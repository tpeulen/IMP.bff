"""The parameter-layer calls the web hosts make on IMP.bff, as one JSON report.

Run the same file natively and under Pyodide and compare the two reports:

    python tools/pyodide/port_checks.py                    # native, prints JSON
    node tools/pyodide/smoke_test.mjs <wheel>              # runs it in Pyodide

What it exercises is what ``chisurf.core.parameter`` (``Parameter`` and
``FittingParameter``) does to its backing ``IMP.bff.GraphPort``: the keyword
constructor with a float64 array value, bounds and their enforcement, the
fixed flag, links and cycle detection, the JSON document round trip, priors,
and registration with the session. Integer ports at and past 2**32 are
included on purpose: wasm32 has a 32-bit ``long`` and ``size_t``, which is
where a truncation would show.

When ``chisurf`` (and ndXplorer, from its ``modules/ndxplorer``) is importable,
the same round trip also runs through
``chisurf.core.fitting.parameter.FittingParameter`` itself, and through what
ndXplorer's overlays build on it: the constants group, a linked constant and a
one-shot curve fit.
"""

import json
import math
import sys
import traceback

import numpy as np

import IMP
import IMP.bff as bff


def _plain(v):
    """A JSON-able form of a port value (numpy scalars and arrays included)."""
    if isinstance(v, np.ndarray):
        return [_plain(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return v if math.isfinite(v) else repr(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def port(value, **kw):
    """The port chisurf's Parameter.__init__ builds."""
    return bff.GraphPort(value=np.atleast_1d(value).astype(np.float64), **kw)


def checks():
    r = {}

    a = port(2.5, name="a", lb=0.0, ub=10.0, is_bounded=True)
    r["float.value"] = _plain(a.value)
    r["float.name"] = a.name
    r["float.bounded"] = a.bounded
    r["float.bounds"] = _plain(a.bounds)
    r["float.lower_upper"] = [a.get_lower_bound(), a.get_upper_bound()]
    a.value = 20.0
    r["float.clipped_high"] = _plain(a.value)
    a.value = -3.0
    r["float.clipped_low"] = _plain(a.value)
    a.bounded = False
    a.value = 42.0
    r["float.unbounded_write"] = _plain(a.value)
    a.bounds = np.array([1.0, 5.0])
    a.bounded = True
    r["float.rebounded"] = _plain(a.value)
    r["float.rebounded_bounds"] = _plain(a.bounds)

    a.fixed = True
    r["fixed.flag"] = a.fixed
    a.value = 3.0
    r["fixed.after_write"] = _plain(a.value)
    a.fixed = False
    a.value = 3.0
    r["fixed.released_write"] = _plain(a.value)

    # links: b follows a; a cycle a -> b is refused before it is made
    b = port(7.0, name="b")
    r["link.before"] = [_plain(a.value), _plain(b.value)]
    r["link.would_cycle_before"] = bool(b.would_create_cycle(a))
    b.link = a
    r["link.is_linked"] = bool(b.is_linked())
    r["link.follows"] = _plain(b.value)
    a.value = 4.5
    r["link.follows_update"] = _plain(b.value)
    r["link.target_name"] = b.link.name
    r["link.would_cycle"] = bool(a.would_create_cycle(b))
    r["link.unlink"] = bool(b.unlink())
    r["link.after_unlink"] = [_plain(b.value), bool(b.is_linked())]

    # a chain of three, then a long chain -- the propagation walk
    c = port(0.0, name="c")
    c.link = b
    b.link = a
    a.value = 1.25
    r["chain3"] = [_plain(x.value) for x in (a, b, c)]
    head = port(0.5, name="p0")
    prev = head
    for i in range(1, 200):
        p = port(0.0, name=f"p{i}")
        p.link = prev
        prev = p
    head.value = 0.75
    r["chain200.tail"] = _plain(prev.value)

    # document round trip, as Parameter's pickle path does it
    d = port(0.3, name="d", lb=0.0, ub=1.0, is_bounded=True)
    d.fixed = True
    d.prior = {"type": "normal", "mu": 0.3, "sigma": 0.05}
    doc = json.loads(d.get_json())
    doc.pop("link", None)
    r["json.doc"] = _plain(doc)
    e = port(0.0, name="")
    e.read_json(d.get_json())
    r["json.restored"] = {
        "value": _plain(e.value), "bounds": _plain(e.bounds),
        "fixed": e.fixed, "bounded": e.bounded, "name": e.name,
        "prior": e.prior,
    }

    # the session registry Parameter adds every port to
    s = bff.get_session()
    s.add_port(d)
    s.add_port(d)
    r["session.type"] = type(s).__name__

    # integer ports: Python ints are exact int64 on the C++ side
    for label, v in [("int.small", 7), ("int.2_31", 2**31), ("int.2_32p1", 2**32 + 1),
                     ("int.2_40", 2**40), ("int.2_53p1", 2**53 + 1),
                     ("int.neg_2_62", -(2**62))]:
        q = bff.GraphPort(value=v)
        r[label] = [_plain(q.value), q.type_name]
    q = bff.GraphPort(value=np.array([1, 2**33, -(2**40)], dtype=np.int64))
    r["int.vector"] = [_plain(q.value), str(np.asarray(q.value).dtype)]
    q = bff.GraphPort(value=True)
    r["bool.scalar"] = [q.value, q.type_name]

    # vector ports: the managed numpy view
    v = port(np.linspace(0.0, 1.0, 5), name="v")
    r["vector.value"] = _plain(v.value)
    v.value = np.arange(1000, dtype=np.float64)
    r["vector.len_sum"] = [len(v.value), float(np.sum(v.value))]
    r["vector.current_size"] = int(v.current_size())
    return r


def fitting_parameter_checks():
    """The same round trip through chisurf's own FittingParameter."""
    from chisurf.core.fitting.parameter import FittingParameter

    r = {}
    x = FittingParameter(value=2.0, name="x", lb=0.0, ub=5.0, bounds_on=True)
    y = FittingParameter(value=9.0, name="y")
    r["fp.value"] = x.value
    r["fp.bounds"] = _plain(list(x.bounds))
    r["fp.bounds_on"] = x.bounds_on
    x.value = 8.0
    r["fp.clipped"] = x.value
    x.fixed = True
    r["fp.fixed"] = x.fixed
    x.fixed = False
    y.link = x
    r["fp.linked"] = [y.is_linked, y.value]
    x.value = 1.5
    r["fp.link_follows"] = y.value
    y.link = None
    r["fp.unlinked"] = [y.is_linked, y.value]
    state = x.get_state()
    r["fp.state"] = _plain(state)
    z = FittingParameter(value=0.0, name="z")
    z.set_state(state)
    r["fp.set_state"] = [z.value, z.fixed, _plain(list(z.bounds)), z.bounds_on]
    r["fp.port_type"] = type(x._port).__name__
    return r


def ndxplorer_checks():
    """What ndXplorer's overlays do: constants as a parameter group, a constant
    linked to a curve parameter, and a curve fit through chisurf's ParseModel."""
    from chisurf.core.fitting.parameter import FittingParameter
    from ndxplorer.analysis.curve_fit import fit_equation_to_marginal
    from ndxplorer.core.constants_group import (build_constants_group,
                                                group_to_value_dict)

    r = {}
    g = build_constants_group({"Bg": 1.5, "gG/gR": 0.8, "tauD0": 4.0})
    r["ndx.constants"] = dict(group_to_value_dict(g))
    r["ndx.constants_fixed"] = [p.fixed for p in g.parameters_all]
    master = FittingParameter(value=0.95, name="gamma")
    g.parameters_all_dict["gG/gR"].link = master
    master.value = 0.9
    r["ndx.constant_linked"] = dict(group_to_value_dict(g))

    # a noiseless Gaussian marginal, so both builds must land on the same numbers
    x = np.linspace(-1.0, 2.0, 61)
    counts = 120.0 * np.exp(-((x - 0.4) ** 2) / (2 * 0.25 ** 2)) + 3.0
    res = fit_equation_to_marginal("A*exp(-(x-m)**2/(2*s**2)) + c",
                                   {"A": 80.0, "m": 0.2, "s": 0.4, "c": 1.0},
                                   x, counts)
    r["ndx.curve_fit.ok"] = res.ok
    r["ndx.curve_fit.message"] = res.message
    r["ndx.curve_fit.params"] = {k: round(float(v), 6) for k, v in res.params.items()}
    return r


def run(with_chisurf=True):
    report = {
        "build": bff.get_build(),
        "imp_path": list(getattr(IMP, "__path__", [])),
        "version": bff.get_module_version(),
    }
    ok = True
    try:
        report["port"] = checks()
    except Exception:
        ok = False
        report["port_error"] = traceback.format_exc()
    if with_chisurf:
        try:
            report["fitting_parameter"] = fitting_parameter_checks()
        except Exception:
            ok = False
            report["fitting_parameter_error"] = traceback.format_exc()
        try:
            report["ndxplorer"] = ndxplorer_checks()
        except Exception:
            ok = False
            report["ndxplorer_error"] = traceback.format_exc()
    report["ok"] = ok
    return report


if __name__ == "__main__":
    rep = run(with_chisurf="--no-chisurf" not in sys.argv)
    print(json.dumps(rep, indent=1, sort_keys=True))
    sys.exit(0 if rep["ok"] else 1)
