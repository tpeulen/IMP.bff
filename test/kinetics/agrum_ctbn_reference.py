"""aGrUM's CTBN amalgamation and exact inference, transcribed without pyagrum.

The A/B reference for `IMP.bff.KineticNetwork` (okf/prds/prd-150.md). pyagrum
is not installed in any env here and `CIM.py` imports it at module level, so the
computation is transcribed as literally as a Tensor-free rewrite allows, with
line references into ../chisurf/junk/aGrUM at 9f2905b60
(wrappers/pyagrum/pyLibs/ctbn/):

* a CIM is a tensor over variables named `<V>#i` (from state), `<V>#j` (to
  state) and bare parent names -- CIM.py 283-316 (varI/varJ/isParent);
* `CTBN.add` creates `CIM().add(v_j).add(v_i)`, `addArc` adds the parent to the
  child's CIM -- CTBN.py add/addArc;
* the diagonal is the negative sum of the row (from-state) off-diagonals --
  CTBNGenerator.py 84-110 (randomCIMs);
* `amalgamate` -- CIM.py 421-499, its four cases verbatim;
* `toMatrix` -- CIM.py 341-383: names sorted, rows the `#i` instantiation,
  columns `#j`, and an aGrUM Instantiation increments its FIRST variable
  fastest (src/agrum/base/multidim/instantiation.cpp 299-324), so the joint
  index is little-endian over the sorted names;
* `SimpleInference.makeInference(t)` -- CTBNInference.py 97-118: expm(t Q),
  uniform initial distribution over the `#i` variables, sum out the `#i`;
  `posterior(v)` sums in `v#j`.

Deliberately NOT importing IMP.bff (prototyping rule, tpeulen 2026-09-14): this
is the reference, the C++ is what is checked against it. It lives beside the
test rather than in prototypes/ (git-ignored) so the fixture stays reproducible.

Run with the arm64 env's python; writes
test/kinetics/data/ctbn_agrum_fixture.json.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

DELIMITER = "#"


def var_i(name):
    return f"{name}{DELIMITER}i"


def var_j(name):
    return f"{name}{DELIMITER}j"


def is_parent(name):
    # CIM.isParent: a name not ending in "#?" is a parent.
    return len(name) < 2 or name[-2] != DELIMITER


def radical(name):
    return name if is_parent(name) else name[:-2]


class Cim:
    """A tensor over named variables: `sizes` maps name -> domain size,
    `order` keeps aGrUM's insertion order, `values` maps a full assignment
    (a tuple of (name, value) sorted by name) to a float, default 0."""

    def __init__(self):
        self.order: list[str] = []
        self.sizes: dict[str, int] = {}
        self.values: dict[tuple, float] = {}

    def add(self, name, size):
        if name not in self.sizes:  # Tensor.contains
            self.order.append(name)
            self.sizes[name] = size
        return self

    @property
    def bases(self):
        return [n for n in self.order if not is_parent(n)]

    @property
    def parents(self):
        return [n for n in self.order if is_parent(n)]

    def nbr_dim(self):
        return len(self.order)

    @staticmethod
    def key(assignment):
        return tuple(sorted(assignment.items()))

    def __getitem__(self, assignment):
        a = {n: assignment[n] for n in self.order}
        return self.values.get(self.key(a), 0.0)

    def __setitem__(self, assignment, value):
        a = {n: assignment[n] for n in self.order}
        self.values[self.key(a)] = float(value)

    def instantiations(self):
        names = self.order
        for combo in itertools.product(*[range(self.sizes[n]) for n in names]):
            yield dict(zip(names, combo))

    def is_im(self):
        return all(not is_parent(n) for n in self.order)

    # CIM.py 421-499
    def amalgamate(self, cim_y: "Cim") -> "Cim":
        cim_x = self
        if cim_x.nbr_dim() == 0:
            return cim_y
        if cim_y.nbr_dim() == 0:
            return cim_x
        s_x = {radical(n) for n in cim_x.order if not is_parent(n)}
        s_y = {radical(n) for n in cim_y.order if not is_parent(n)}
        amal = Cim()
        for cim in (cim_x, cim_y):
            for n in cim.bases:
                amal.add(n, cim.sizes[n])
        p_x_in_y, p_y_in_x = set(), set()
        for n in cim_x.parents:
            if n not in s_y:
                amal.add(n, cim_x.sizes[n])
            else:
                p_y_in_x.add(n)
        for n in cim_y.parents:
            if n not in s_x:
                amal.add(n, cim_y.sizes[n])
            else:
                p_x_in_y.add(n)
        for i in amal.instantiations():
            i_x = {n: i[n] for n in cim_x.order if n in i}
            i_y = {n: i[n] for n in cim_y.order if n in i}
            for v in p_x_in_y:
                i_y[v] = i[var_i(v)]  # iY.chgVal(v, iX[CIM.varI(v)])
            for v in p_y_in_x:
                i_x[v] = i[var_i(v)]
            d_x = all(i[var_i(v)] == i[var_j(v)] for v in s_x)
            d_y = all(i[var_i(v)] == i[var_j(v)] for v in s_y)
            if d_x and d_y:
                amal[i] = cim_x[i_x] + cim_y[i_y]
            elif d_y:
                amal[i] = cim_x[i_x]
            elif d_x:
                amal[i] = cim_y[i_y]
            else:
                amal[i] = 0.0
        return amal

    def _little_endian(self, names):
        """Assignments of `names` in aGrUM Instantiation order (first fastest)."""
        out = []
        for combo in itertools.product(*[range(self.sizes[n]) for n in reversed(names)]):
            out.append(dict(zip(reversed(names), combo)))
        return out

    # CIM.py 341-383
    def to_matrix(self):
        if not self.is_im():
            raise ValueError("The cim is conditionnal.")
        names = sorted(self.order)
        i_names = [n for n in names if n[-1] == "i"]
        j_names = [n for n in names if n[-1] != "i"]
        rows = []
        for a_i in self._little_endian(i_names):
            line = []
            for a_j in self._little_endian(j_names):
                line.append(self[{**a_i, **a_j}])
            rows.append(line)
        return np.array(rows), [radical(n) for n in i_names]


class Ctbn:
    def __init__(self):
        self.names: list[str] = []
        self.sizes: dict[str, int] = {}
        self.cims: dict[str, Cim] = {}

    def add(self, name, size):
        self.names.append(name)
        self.sizes[name] = size
        self.cims[name] = Cim().add(var_j(name), size).add(var_i(name), size)

    def add_arc(self, parent, child):
        self.cims[child].add(parent, self.sizes[parent])

    def set_rate(self, name, source, target, rate, parents=None):
        """Off-diagonal rate source -> target; the diagonal as randomCIMs sets it."""
        cim = self.cims[name]
        a = dict(parents or {})
        a[var_i(name)] = source
        a[var_j(name)] = target
        cim[a] = rate
        diag = dict(a)
        diag[var_j(name)] = source
        exit_rate = sum(
            cim[{**a, var_j(name): j}] for j in range(self.sizes[name]) if j != source
        )
        cim[diag] = -exit_rate

    def joint(self):
        q = Cim()
        for n in self.names:  # SimpleInference.makeInference: for nod in nodes()
            q = q.amalgamate(self.cims[n])
        return q

    def simple_inference(self, t):
        """CTBNInference.py 97-138: posteriors of every variable at time t."""
        matrix, order = self.joint().to_matrix()
        p = expm(t * matrix)
        n = matrix.shape[0]
        t0 = np.full(n, 1.0 / n)
        joint_j = t0 @ p  # rows are from-states: a row vector propagates
        sizes = [self.sizes[v] for v in order]
        # little-endian over `order`: reshape with the first variable last
        tensor = joint_j.reshape(list(reversed(sizes)))
        post = {}
        for k, v in enumerate(order):
            axis = len(order) - 1 - k
            other = tuple(a for a in range(len(order)) if a != axis)
            post[v] = tensor.sum(axis=other).tolist()
        return post


def conformation_photophysics():
    """Conformation C (open, closed) x photophysics P (S0, S1, T), rates in 1/us.

    The PET/PIFE-type FCS scheme: conformational exchange is independent of
    the dye's state; S1 decays faster in the closed (quenched) conformation, so
    P's CIM is conditioned on C. Excitation and triplet rates are
    rhodamine_3state.json's (chisurf fcs_saturation_calc/schemes)."""
    net = Ctbn()
    net.add("C", 2)
    net.add("P", 3)
    net.add_arc("C", "P")
    net.set_rate("C", 0, 1, 0.8)  # open -> closed
    net.set_rate("C", 1, 0, 0.3)  # closed -> open
    for c, k_s1 in ((0, 250.0), (1, 1000.0)):
        net.set_rate("P", 0, 1, 5.0, {"C": c})  # S0 -> S1, excitation
        net.set_rate("P", 1, 0, k_s1, {"C": c})  # S1 -> S0, quenched when closed
        net.set_rate("P", 1, 2, 2.5, {"C": c})  # S1 -> T
        net.set_rate("P", 2, 0, 0.5, {"C": c})  # T -> S0
    return net


def cyclic_three_variables():
    """A (2) <-> B (3), B -> D (2): both amalgamation branches that look up a
    parent in the other CIM's from-state (pXinY and pYinX), and a cycle."""
    rng = np.random.default_rng(20260917)
    net = Ctbn()
    net.add("A", 2)
    net.add("B", 3)
    net.add("D", 2)
    net.add_arc("B", "A")
    net.add_arc("A", "B")
    net.add_arc("B", "D")
    for name, parent in (("A", "B"), ("B", "A"), ("D", "B")):
        for pv in range(net.sizes[parent]):
            for s in range(net.sizes[name]):
                for t in range(net.sizes[name]):
                    if s != t:
                        net.set_rate(name, s, t, float(rng.uniform(0.1, 3.0)), {parent: pv})
    return net


def describe(net, t):
    matrix, order = net.joint().to_matrix()
    rates = []
    for name in net.names:
        cim = net.cims[name]
        for a in cim.instantiations():
            if a[var_i(name)] != a[var_j(name)]:
                rates.append(
                    {
                        "variable": name,
                        "source": a[var_i(name)],
                        "target": a[var_j(name)],
                        "parents": {p: a[p] for p in cim.parents},
                        "rate": cim[a],
                    }
                )
    return {
        "variables": [{"name": n, "states": net.sizes[n]} for n in net.names],
        "arcs": [[p, n] for n in net.names for p in net.cims[n].parents],
        "rates": rates,
        "agrum_matrix_row_from_col_to": matrix.tolist(),
        "agrum_index_order_little_endian": order,
        "inference_time": t,
        "agrum_posteriors": net.simple_inference(t),
    }


def main():
    fixture = {
        "source": "aGrUM 9f2905b60 wrappers/pyagrum/pyLibs/ctbn, transcribed by "
        "test/kinetics/agrum_ctbn_reference.py",
        "schemes": {
            "conformation_photophysics": describe(conformation_photophysics(), 0.05),
            "cyclic_three_variables": describe(cyclic_three_variables(), 0.7),
        },
    }
    out = Path(__file__).resolve().parent / "data" / "ctbn_agrum_fixture.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=1) + "\n")
    for name, s in fixture["schemes"].items():
        m = np.array(s["agrum_matrix_row_from_col_to"])
        print(name, m.shape, "max |row sum|", np.abs(m.sum(axis=1)).max(), s["agrum_posteriors"])
    print("wrote", out)


if __name__ == "__main__":
    main()
