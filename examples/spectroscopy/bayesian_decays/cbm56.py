"""CBM56 variant 3: a real measurement, read and described.

Fourteen TAC histograms in `data/CBM56/Var3_TAC`, measured by Alex on a Paris
spectrometer under pulsed interleaved excitation and sent by Suren. Everything
in the `bayesian_decays` series before this was fitted to data the model itself
produced; this is the first measurement from an instrument.

What the folder holds
---------------------
Each file is **1024 numbers: two 512-channel histograms**, the two polarisations
of one detector pair. The pair is named in the file name -- `10ps` is green
(Paris channels 1 and 0), `32ps` is red (channels 3 and 2) -- and `ps` is
parallel and perpendicular, in that order.

* `Var3/Var3_BID {D0,A0,DA}_*.dat` -- the three burst cuts
* `IRF/h20{,_2}_*.dat` -- two water measurements, the instrument response
* `Rh110_thick/rhd110{,_2}_*.dat` -- a reference dye, for control
* `Paris_x64 info.txt` -- the calibration

The geometry is already described
---------------------------------
Three samples, two interleaved pulses, four detectors: that is
`experiment.mfd_pie()`. `experiment()` below fills that description in with this
spectrometer's numbers rather than restating it.

The one thing to know before reading anything else: **the TAC range is longer
than the laser period** -- 32.768 ns against 31.249 ns -- so the last two dozen
channels are outside the period, no photon can arrive in them, and they are the
cleanest background estimate in the file.

Author: written for the bff examples, 2026-09-14.
See `okf/prd-cbm56-real-data.md` in the ucfret repository for the plan.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

__all__ = ['data_dir', 'read_calibration', 'load', 'experiment', 'pulse_windows',
           'reference_dye', 'g_from_reference', 'apply_calibration_priors',
           'period_from_data', 'background', 'count_rates', 'pulse_positions',
           'DETECTORS', 'responses', 'pulse_offset_channels', 'histograms',
           'CAL', 'SAMPLES', 'PAIRS', 'POLARISATIONS']

#: the three burst cuts, and what each is
SAMPLES = {'D0': 'donor only', 'A0': 'acceptor only', 'DA': 'both, the FRET sample'}
#: the two detector pairs, by the Paris channel numbers in the file name
PAIRS = {'10ps': ('green', (1, 0)), '32ps': ('red', (3, 2))}
POLARISATIONS = ('parallel', 'perpendicular')


def data_dir(root=None) -> Path:
    """`data/CBM56/Var3_TAC`, found from this file or from `root`."""
    if root is not None:
        p = Path(root)
        return p if p.name == 'Var3_TAC' else p / 'Var3_TAC'
    here = Path(__file__).resolve()
    for parent in here.parents:
        c = parent / 'data' / 'CBM56' / 'Var3_TAC'
        if c.is_dir():
            return c
    raise FileNotFoundError('data/CBM56/Var3_TAC not found above ' + str(here))


# --------------------------------------------------------------------------
# the calibration
# --------------------------------------------------------------------------

def read_calibration(path=None) -> dict:
    """`Paris_x64 info.txt`, as a dictionary.

    The file is tab-separated `section, key, value` with the sections repeating
    for the three colour fits, so the colour blocks are kept apart by the
    `Green fit parameters` / `Red fit parameters` / `Yellow fit parameters`
    markers that separate them. Numbers are written in engineering notation
    (`31.24922561E+0`) and are converted; everything else is kept as text.
    """
    p = Path(path) if path is not None else data_dir() / 'Paris_x64 info.txt'
    out, colour = {}, None
    for line in p.read_text(errors='replace').splitlines():
        parts = [x.strip() for x in line.split('\t') if x.strip()]
        if not parts:
            continue
        #: the colour blocks are separated by a line whose only content is
        #: `Green fit parameters` (behind an empty first field), and the three
        #: blocks repeat the same keys, so this marker is what keeps them apart
        if parts[-1].endswith('fit parameters'):
            colour = parts[-1].split()[0].lower()
            out.setdefault(colour, {})
            continue
        if len(parts) < 2:
            continue
        key, value = parts[-2].rstrip(':'), parts[-1]
        try:
            v = float(value)                 # Python reads Paris's `31.24922561E+0`
        except ValueError:
            v = value
        target = out[colour] if (colour and parts[0] == 'colour fit parameters') else out
        target[key] = v
    return out


#: the numbers this module uses, read once and kept, so that a notebook can
#: print them beside the data rather than having them buried in a call
def _cal():
    c = read_calibration()
    g = c.get('green', {}); r = c.get('red', {}); y = c.get('yellow', {})
    return dict(
        dt=float(c.get('TAC cal.', 0.064)),                    # ns per channel
        period=float(c.get('Laser repetition time', 31.24922561)),   # ns
        n_channels=int(c.get('maxTAC', 511)) + 1,
        tac_range=float(c.get('TAC Range', 32.768)),
        g_green=float(g.get('G-factor', float('nan'))),
        g_red=float(r.get('G-factor', float('nan'))),
        g_yellow=float(y.get('G-factor', float('nan'))),
        scatter_green_hz=float(g.get('Scatter Countrate [Hz]', float('nan'))),
        scatter_red_hz=float(r.get('Scatter Countrate [Hz]', float('nan'))),
        scatter_yellow_hz=float(y.get('Scatter Countrate [Hz]', float('nan'))),
        l1=float(c.get('Japan. corr. factors.[0]', 0.0175)),
        l2=float(c.get('Japan. corr. factors.[1]', 0.0526)),
        window_green=(int(g.get('from', 13)), int(g.get('to', 250))),
        window_red=(int(r.get('from', 40)), int(r.get('to', 250))),
        window_yellow=(int(y.get('from', 258)), int(y.get('to', 425))),
        tau_donor=float(c.get('tau D', 4.0)),
        tau_da=float(c.get('tau DA', 2.0)),
        crosstalk=float(c.get('Crosstalk', 0.07)),
        direct_excitation=float(c.get('Direct Excitation', 0.01)),
        qy_donor=float(c.get('QY', 0.7)), qy_acceptor=float(c.get('QY R', 0.3)),
        raw=c)


CAL = None          # filled by `load`, so importing this module reads nothing


# --------------------------------------------------------------------------
# the histograms
# --------------------------------------------------------------------------

def _read_pair(path, n_channels):
    a = np.loadtxt(path, dtype=float)
    if a.size != 2 * n_channels:
        raise ValueError(f'{path.name}: {a.size} values, expected {2 * n_channels}')
    return {'parallel': a[:n_channels], 'perpendicular': a[n_channels:]}


def load(root=None) -> dict:
    """Every histogram in the folder, with the calibration and the time axis.

    Returns `sample`, `irf` and `reference` dictionaries keyed by
    `(name, colour, polarisation)`, the acquisition time of each burst cut in
    seconds (it is in the file name, which is the only place it appears), and
    the calibration.
    """
    global CAL
    d = data_dir(root)
    cal = _cal(); CAL = cal
    n = cal['n_channels']
    out = dict(cal=cal, dir=d,
               t=(np.arange(n) + 0.5) * cal['dt'],
               sample={}, irf={}, reference={}, seconds={})
    for f in sorted((d / 'Var3').glob('*.dat')):
        m = re.search(r'BID (\w+)_([\d.]+) s_(\w+)\.dat$', f.name)
        if not m:
            continue
        samp, secs, pair = m.group(1), float(m.group(2)), m.group(3)
        colour = PAIRS[pair][0]
        out['seconds'][samp] = secs
        for pol, h in _read_pair(f, n).items():
            out['sample'][(samp, colour, pol)] = h
    for f in sorted((d / 'IRF').glob('*.dat')):
        m = re.search(r'(h20(?:_2)?)_(\d-\d) ps', f.name)
        if not m:
            continue
        which = m.group(1)
        colour = 'green' if m.group(2) == '1-0' else 'red'
        for pol, h in _read_pair(f, n).items():
            out['irf'][(which, colour, pol)] = h
    for f in sorted((d / 'Rh110_thick').glob('*.dat')):
        m = re.search(r'(rhd110(?:_2)?)_(\d-\d) ps', f.name)
        if not m:
            continue
        colour = 'green' if m.group(2) == '1-0' else 'red'
        for pol, h in _read_pair(f, n).items():
            out['reference'][(m.group(1), colour, pol)] = h
    return out


# --------------------------------------------------------------------------
# what the histograms say about themselves
# --------------------------------------------------------------------------

def pulse_windows(cal) -> dict:
    """The channel ranges the two pulses occupy, and the part of the TAC range
    that lies beyond the laser period.

    **The region beyond the period is not a background estimate.** It looked
    like one -- 31.249 ns of period over a 32.768 ns TAC range leaves two dozen
    channels no photon can reach -- but every histogram in this folder is
    identically zero there, because the electronics cannot record it at all.
    What it is instead is an independent confirmation of the period: the last
    channel carrying a count is 487 or 488, and the calibration file's
    31.24922561 ns is 488.27 channels.
    """
    n, dt, period = cal['n_channels'], cal['dt'], cal['period']
    beyond = int(np.ceil(period / dt))
    return dict(green=cal['window_green'], sensitised=cal['window_red'],
                red=cal['window_yellow'], beyond_period=(beyond, n),
                channels_beyond=n - beyond)


def period_from_data(loaded) -> dict:
    """The laser period, read off the histograms rather than off the file.

    The last channel that carries a count is where the period ends, because the
    TAC range is longer than the period and the electronics stop there.
    """
    last = {}
    for group in ('sample', 'irf', 'reference'):
        for k, h in loaded[group].items():
            nz = np.nonzero(h)[0]
            if len(nz):
                last[k] = int(nz[-1])
    ch = np.array(sorted(last.values()))
    dt = loaded['cal']['dt']
    return dict(last_channel=last, median=float(np.median(ch)),
                ns=float(np.median(ch) * dt), stated=loaded['cal']['period'],
                stated_channels=loaded['cal']['period'] / dt)


def background(h, cal, flat=((0, 13), (440, 481))) -> dict:
    """The background per channel, from the flat parts of the decay.

    Not from beyond the period, which is empty (see `pulse_windows`). The two
    default windows are before the green pulse and at the end of the period, and
    they are quoted SEPARATELY as well as together: they sit at opposite ends of
    the decay, so if they disagree the flat level is not flat and the number is
    not a background.
    """
    h = np.asarray(h)
    segs = [h[a:b] for a, b in flat]
    allseg = np.concatenate(segs)
    return dict(per_channel=float(allseg.mean()),
                sd=float(allseg.std(ddof=1) / max(np.sqrt(len(allseg)), 1)),
                windows=[float(s.mean()) for s in segs],
                n_channels=int(len(allseg)), total=float(allseg.sum()),
                fraction_of_histogram=float(allseg.mean() * len(h) / max(h.sum(), 1)))


def pulse_positions(loaded, split=255) -> dict:
    """Where each pulse sits in each detector, from the water measurements.

    Reported per detector because they do not agree: the green pulse peaks nine
    channels later in the red detectors than in the green ones, and the two
    polarisations of one colour differ by three. That is the colour shift and
    the per-detector timing, and it is why the model carries a shift for every
    detector rather than one for the instrument.
    """
    out = {}
    for k, h in sorted(loaded['irf'].items()):
        g, r = h[:split], h[split:]
        out[k] = dict(green_channel=int(np.argmax(g)), green_counts=float(g.sum()),
                      red_channel=split + int(np.argmax(r)), red_counts=float(r.sum()))
    return out


def count_rates(loaded, sample, colour) -> dict:
    """Counts per second in one burst cut's two polarisations, which is what
    the calibration file quotes its scatter rates in."""
    secs = loaded['seconds'][sample]
    out = {}
    for pol in POLARISATIONS:
        h = loaded['sample'][(sample, colour, pol)]
        out[pol] = float(h.sum()) / secs
    out['total'] = sum(out[p] for p in POLARISATIONS)
    out['seconds'] = secs
    return out


def experiment():
    """This measurement, as an `Experiment`.

    It is `mfd_pie()` with the detector names this spectrometer uses. Nothing
    about the geometry is restated here -- three samples, two interleaved
    pulses, four detectors -- because the description layer already has it; what
    is filled in is which detector is which.
    """
    import experiment as X
    dets = [X.Detector('gp', 'green'), X.Detector('gs', 'green'),
            X.Detector('rp', 'red'), X.Detector('rs', 'red')]
    pol = {'gp': 'parallel', 'gs': 'perpendicular',
           'rp': 'parallel', 'rs': 'perpendicular'}
    samples = [X.Sample('D0', ('donor',)), X.Sample('A0', ('acceptor',)),
               X.Sample('DA', ('donor', 'acceptor'))]
    cal = CAL or _cal()
    green = X.Excitation('green', {'donor': 1.0, 'acceptor': 'EX_AG'}, 0.0)
    red = X.Excitation('red', {'acceptor': 1.0, 'donor': 'EX_DR'},
                       (cal['window_yellow'][0] - cal['window_green'][0]) * cal['dt'])
    chans = [X.Channel(s.name, e, d.name, pol[d.name])
             for s in samples for d in dets for e in ('green', 'red')]
    return X.Experiment(
        name='CBM56 Var3, Paris, 32 MHz PIE',
        samples=samples, excitations=[green, red], detectors=dets,
        channels=chans, interleaved=True,
        notes=f'TAC {cal["dt"]*1000:.0f} ps over {cal["n_channels"]} channels, '
              f'laser period {cal["period"]:.3f} ns; g green {cal["g_green"]:.4f}, '
              f'red {cal["g_red"]:.4f}; l1 {cal["l1"]:.4f}, l2 {cal["l2"]:.4f}',
    ).validate()


# --------------------------------------------------------------------------
# what the model needs: the four detectors, their responses, the data
# --------------------------------------------------------------------------

#: the four physical detectors, and which half of which file each is
DETECTORS = {'gp': ('green', 'parallel'), 'gs': ('green', 'perpendicular'),
             'rp': ('red', 'parallel'), 'rs': ('red', 'perpendicular')}
#: the model's polarisation code for each
KIND = {'gp': 'vv', 'gs': 'vh', 'rp': 'vv', 'rs': 'vh'}


def pulse_offset_channels(loaded, which='h20') -> int:
    """How far the red pulse is behind the green one, in channels, measured in
    the RED detectors -- the only ones that see both.

    Not taken from the calibration file's fit windows: those are where Alex
    chose to fit, which is near the pulses but not at them.
    """
    out = []
    for pol in POLARISATIONS:
        h = loaded['irf'][(which, 'red', pol)]
        out.append(int(np.argmax(h[255:])) + 255 - int(np.argmax(h[:255])))
    return int(round(float(np.mean(out))))


def responses(loaded, which='h20', pad=(15, 150), n=None) -> dict:
    """The instrument response of each detector, at the GREEN pulse's position.

    Which pulse each response is taken from is not a detail. The green
    detectors see only the green pulse, so there is no choice. **The red
    detectors see both, and the green-pulse one is far the worse measured** --
    about 2,250 counts at its peak against a pedestal of 710 per channel, where
    the red-pulse response peaks two orders of magnitude higher. A detector's
    response does not depend on which laser fired, so the red detectors' shape
    is taken from their red pulse and moved back to where the green pulse sits
    in them. That is a choice and it is made here, in the open.

    The window keeps the pedestal inside it: removing the response's own
    background is the fit's job (`irf_bg_<detector>`), not this function's.
    """
    n = int(loaded['cal']['n_channels']) if n is None else int(n)
    off = pulse_offset_channels(loaded, which)
    out, info = {}, {}
    for det, (colour, pol) in DETECTORS.items():
        h = np.asarray(loaded['irf'][(which, colour, pol)], float)
        if colour == 'green':
            peak = int(np.argmax(h[:255])); source = 'the green pulse'
        else:
            peak = int(np.argmax(h[255:])) + 255; source = 'the red pulse, moved back'
        a, b = max(peak - pad[0], 0), min(peak + pad[1], len(h))
        seg = np.zeros(len(h)); seg[a:b] = h[a:b]
        if colour == 'red':
            seg = np.roll(seg, -off)
        r = seg[:n].copy()
        out[det] = r
        flat = float(np.median(h[200:251])) if colour == 'green' else float(np.median(h[430:481]))
        info[det] = dict(peak_channel=peak, source=source, counts=float(r.sum()),
                         peak=float(r.max()), flat_per_channel=flat,
                         background_fraction=float(min(flat * (b - a) / max(r.sum(), 1), 0.95)))
    return out, info, off


def histograms(loaded, n=None) -> dict:
    """The twelve measured histograms keyed the way the model keys them, and
    the mask that excludes the channels beyond the laser period.

    Those channels are structurally empty -- the electronics never reach them --
    so a model that predicts a background there would be fitted against a zero
    that means nothing.
    """
    n = int(loaded['cal']['n_channels']) if n is None else int(n)
    y, mask = {}, {}
    for det, (colour, pol) in DETECTORS.items():
        for samp in ('D0', 'A0', 'DA'):
            h = np.asarray(loaded['sample'][(samp, colour, pol)], float)[:n]
            y[(samp, f'{det}_{KIND[det]}')] = h
            mask[(samp, f'{det}_{KIND[det]}')] = np.ones(n)
    return y, mask


# --------------------------------------------------------------------------
# the model's environment, on this spectrometer's time axis
# --------------------------------------------------------------------------

def environment(n_coef=25, n=488, cache=True, verbose=True):
    """The transfer maps, built on THIS instrument's axis rather than the
    prototype's.

    The maps are functions of the time axis, and the prototype's is a 50 ns
    period at 32 ps because that is the instrument it was written against.
    CBM56 is 31.25 ns at 64 ps. Both are arguments to the builder, so this is a
    parameter change and a cache, not a redesign -- but it is a change that
    could go wrong silently, so `check_axis` compares the two.

    **The axis is the measured one.** 488 channels of exactly 64 ps is 31.232 ns
    against the calibration file's 31.24922561, so the period is 0.055 % short.
    The alternative -- an exact period and channels of 64.035 ps -- puts the same
    error into every lifetime instead. Keeping the TAC calibration exact is the
    better of the two, because the calibration is what a lifetime is measured
    in, and 0.055 % of a wrap-around is smaller than 0.055 % of a lifetime.
    """
    import os, time, torch
    import sys as _sys
    from bd import P
    L = P.load_prototype(threads=4)
    d = P.prototype_dir()
    if str(d) not in _sys.path:
        _sys.path.insert(0, str(d))
    import s80_analytic_stage2 as A, s79_fret_stage2 as S
    import s83_relative_distance as R3, s86_sensitised as S6, s87_amortized_nn as M

    cal = CAL or _cal()
    dt = cal['dt']
    rel, edges = R3.grid(M.N_REL)
    S.TAU_REF = L.TAU_0
    S.R_GRID = S.R0_FOERSTER * rel
    ck = d / 'ckpt' / 'homog'
    ck.mkdir(parents=True, exist_ok=True)
    f = ck / f'cbm56_env_{n}_{M.N_REL}.pt'
    if cache and f.exists():
        E = torch.load(f, weights_only=False)
        if verbose:
            print(f'  maps loaded from {f.name}')
    else:
        t0 = time.time()
        E = A.build(n=n, period=n * dt, rebin=False, acceptor_grid=True)
        if verbose:
            print(f'  transfer maps on the CBM56 axis in {time.time() - t0:.0f} s')
        if cache:
            torch.save(E, f)
    f2 = ck / f'cbm56_rho_{n}_{len(M.RHO_GRID)}.pt'
    if cache and f2.exists():
        E['S_rho'] = torch.load(f2, weights_only=False)
    else:
        t0 = time.time()
        b = E['basis']; keep = b.B; b.B = E['B_full']
        try:
            E['S_rho'] = S.rot_maps(E['dec'], b, E['T'], E['tau_c'], M.RHO_GRID)
        finally:
            b.B = keep
        if verbose:
            print(f'  rotational maps in {time.time() - t0:.0f} s')
        if cache:
            torch.save(E['S_rho'], f2)
    E['rho'] = M.RHO_GRID
    f3 = ck / f'cbm56_rhoa_{n}_{len(M.RHO_A_GRID)}.pt'
    if cache and f3.exists():
        d3 = torch.load(f3, weights_only=False)
    else:
        t0 = time.time()
        b = E['basis']; keep = b.B; b.B = E['B_full']
        try:
            sag, adr = [], []
            for v in M.RHO_A_GRID:
                _, sr = S.sens_maps_grid(E['dec'], b, E['T'], E['tau_c'], S.R_GRID,
                                         S.R0_FOERSTER, S.TAU_A_GRID, rho_a=float(v))
                _, ar = S.direct_maps(E['dec'], b, S.TAU_A_GRID, rho_a=float(v))
                sag.append(sr); adr.append(ar)
        finally:
            b.B = keep
        d3 = dict(S_Ag_rot_r=torch.stack(sag), A_dir_rot_r=torch.stack(adr))
        if verbose:
            print(f'  acceptor rotational maps in {time.time() - t0:.0f} s')
        if cache:
            torch.save(d3, f3)
    E.update(d3)
    E['rho_a'] = M.RHO_A_GRID
    E['rel'], E['edges'] = rel, edges
    E['homogeneous'] = True; E['tau_0'] = float(L.TAU_0)
    #: a nominal response, only so that anything asking for a width default has
    #: one; the responses actually used are measured
    E['instrument'] = ('CBM56 water', 0.15, 0.0, 0.0)
    spl = S6.pspline_basis(M.N_REL, n_coef=n_coef)
    return E, np.asarray(rel), spl, L


# --------------------------------------------------------------------------
# the model, assembled
# --------------------------------------------------------------------------

#: Rhodamine 110 in water: 4.00 ns (Magde, Rojas & Seybold, Photochem. Photobiol.
#: 75, 327, 2002), the prior median of `irf_tauref_<detector>` when the reference
#: dye's decay is the response
TAU_RH110_NS = 4.00


def pulse_bins(E, peak_channels, offset, growth=1.05):
    """THE ADAPTIVE BINNING, restarted at each pulse (tpeulen, 2026-09-14:
    "where is the adaptive binning?").

    The prototype's `widening_bins` (s80) keeps full resolution across a
    response's rise and peak and widens the bins geometrically through the
    tail, so that each bin holds a comparable number of photons -- adding
    adjacent channels of a Poisson histogram is exactly another Poisson
    histogram, so this costs nothing statistically and only decides where
    resolution is spent. It knows ONE peak, and `cbm56.environment` therefore
    built this axis with `rebin=False`: an interleaved histogram has two rises,
    and bins that have widened to twenty channels by the red pulse would smear
    its edge. S15 solved that for the simulated PIE experiment by running the
    scheme once per pulse window and concatenating; this does the same on the
    measured axis. The maps stay projected at full resolution; only the counts
    and the basis are rebinned (`basis_from_irf` applies `E['R']`).
    """
    import torch
    import s80_analytic_stage2 as A
    n, dt = int(E['n']), float(E['dt'])
    pk = float(np.median(list(peak_channels)))
    e1 = A.widening_bins(offset, dt, pk * dt + 0.4, growth, t_rise=max(pk * dt - 0.35, 0.0))
    e2 = A.widening_bins(n - offset, dt, pk * dt + 0.4, growth, t_rise=max(pk * dt - 0.35, 0.0))
    edges = list(e1) + [offset + e for e in e2[1:]]
    edges = sorted(set(int(e) for e in edges if 0 <= e <= n))
    if edges[0] != 0:
        edges = [0] + edges
    if edges[-1] != n:
        edges.append(n)
    Rm = torch.zeros(len(edges) - 1, n, dtype=torch.float64)
    for i in range(len(edges) - 1):
        Rm[i, edges[i]:edges[i + 1]] = 1.0
    return dict(R=Rm, edges=np.array(edges),
                tc=np.array([(edges[i] + edges[i + 1] - 1) / 2 * dt for i in range(len(edges) - 1)]),
                wid=np.array([(edges[i + 1] - edges[i]) * dt for i in range(len(edges) - 1)]),
                win_green=np.array([1.0 if edges[i] < offset else 0.0 for i in range(len(edges) - 1)]))


def rl_deconvolve(R, tau, dt, n_iter=500, eps=1e-12, f0=None):
    """The instrument response from a reference dye's decay by Richardson-Lucy
    deconvolution with a single-exponential kernel of lifetime `tau`
    (Richardson, J. Opt. Soc. Am. 62, 55, 1972; Lucy, Astron. J. 79, 745,
    1974) -- the 'decon' route of tpeulen's prompt 409; tttrlib carries the
    same algorithm for images (`richardson_lucy_2d`), not for a 1-D decay.

    WHY NOT THE DELTA-FUNCTION IDENTITY.  The model carries Zuker's exact
    method (`basis_from_irf(..., tau_ref)`), and on this axis it fails for a
    4 ns reference: the discrete periodic columns do not satisfy the
    continuous identity, the corrected columns of every lifetime shorter than
    about 0.94 tau_ref have NEGATIVE sums, and the unit-sum normalisation turns
    them into 1e297 (measured 2026-09-14; the identity's own check reaches
    only 3e-2 even for a 0.09 ns reference here). Richardson-Lucy's
    multiplicative updates keep the estimate non-negative.

    Measured on this axis with an analytic response convolved with 4.0 ns:
    the width is recovered exactly by 200 iterations, the shape to 14 % of the
    peak at 200 and 5 % at 1000; a Poisson draw at the Rh110 count level
    (1.2e6) changes neither. The result is then a MEASURED response like any
    other: its background and shift stay nuisances of the fit.
    """
    R = np.asarray(R, float); N = len(R); j = np.arange(N)
    k = np.exp(-j * dt / tau); k /= k.sum()
    K = np.fft.rfft(k); Kc = np.conj(K)
    conv = lambda x, F: np.fft.irfft(np.fft.rfft(x) * F, N)
    #: STARTED AT THE WATER RESPONSE when one is given -- "the experimental
    #: irf will only serve as prior" (prompt 409).  From a flat start the
    #: iteration is nowhere near converged for a 4 ns kernel at any count this
    #: measurement can afford (measured: a cliff and no rising edge at 200
    #: iterations); from the measured response it refines what the water
    #: measurement got wrong and keeps what it got right.
    f = np.full(N, R.sum() / N) if f0 is None else np.maximum(np.asarray(f0, float), 0.0) * (R.sum() / max(np.sum(f0), 1e-300))
    for _ in range(int(n_iter)):
        est = conv(f, K)
        f = f * conv(R / np.maximum(est, eps), Kc)
    return f


def model(loaded=None, n_coef=25, which='h20', verbose=True,
          samples=('D0', 'A0', 'DA'), detectors=None, irf='h20', rebin=False, growth=1.05,
          rl_iterations=500, irf_conv_stop=None):
    """Everything the fit needs: the maps on this axis, the measured responses,
    the twelve histograms, and the graph.

    The geometry is the one `experiment.py` calls interleaved: each histogram
    holds both pulse windows, so every (sample, detector) appears twice among
    the physics channels -- once for each pulse -- and the data live on the
    green-pulse key with the red-pulse partner summed into the same mean.
    """
    import torch
    irf_kind = irf                 # `irf` is rebound to the responses below
    d = loaded or load()
    cal = d['cal']
    E, rel, spl, L = environment(n_coef=n_coef, verbose=verbose)
    n = int(E['n'])
    irf, info, off = responses(d, which=which, n=n)
    irf_h20 = {k: v.copy() for k, v in irf.items()}
    y_np, mask_np = histograms(d, n=n)

    E = dict(E)
    E['pulse_alias'] = {'g2p': 'gp', 'g2s': 'gs', 'r2p': 'rp', 'r2s': 'rs'}
    E['pulse_offset'] = {a: off * cal['dt'] for a in E['pulse_alias']}
    E['pie_full'] = True
    #: chisurf's trick for a bad response (prompt 413): convolve the lifetimes
    #: with the pulse only -- `irf_conv_stop` ns after the peak -- and let the
    #: scatter column carry the measured tail
    E['irf_conv_stop'] = None if irf_conv_stop is None else float(irf_conv_stop)
    #: REBIN OFF BY DEFAULT, for now.  The widening bins are right in principle
    #: and cost nothing statistically, and today they break the walk: the same
    #: donor-only fit that converges unbinned to D/dof 1.219 (reference 1.116
    #: +- 0.050, z +2.0 -- the first real measurement to pass Rule 0,
    #: 2026-09-14) runs on the binned axis to a "converged" mode with the green
    #: response shifted by +15.99 ns onto the red pulse, or to a background
    #: fraction of 4.79. That is the start and the step, not the bins (R3 of
    #: okf/prd-real-data-fast.md), and until it is fixed the bins stay opt-in.
    if rebin:
        rb = pulse_bins(E, [info[det]['peak_channel'] for det in DETECTORS if det[0] == 'g'], off, growth)
        E['R'] = rb['R']; E['tc'] = rb['tc']; E['wid'] = rb['wid']; E['n_bin'] = len(rb['tc'])
        win_g = rb['win_green']
        t_axis = rb['tc']
    else:
        E['R'] = None; E['n_bin'] = n
        win_g = np.zeros(n); win_g[:off] = 1.0
        t_axis = np.arange(n) * cal['dt']
    E['win_green'] = L.tt(win_g); E['win_red'] = L.tt(1.0 - win_g)

    dets = tuple(DETECTORS) if detectors is None else tuple(detectors)
    #: WHAT THE RESPONSE IS (tpeulen, 2026-09-14: "there seem to be issues with
    #: the IRF. for the donor, use Rh110 as additional information for the IRF,
    #: eg, either by decon (see tttrlib) or by a skewed gaussian, the
    #: experimental irf will only serve as prior").  The water measurement has a
    #: tail at 1e-3 of its peak out to 8 ns that no background correction
    #: removes, and a tail in the response is a long lifetime to a fit.
    #:
    #:   'h20'       the water measurement, background a nuisance (the default)
    #:   'rh110'     the reference dye's own decay as the response, its lifetime
    #:               a nuisance -- the delta-function convolution method, exact,
    #:               no deconvolution (Zuker et al. 1985; basis_from_irf)
    #:   'analytic'  a skewed Gaussian with free position, width and skew; the
    #:               water measurement sets only their priors
    ref_dets = ()
    if irf_kind == 'rh110':
        #: the reference dye's decay, its own flat background removed, then the
        #: 4.00 ns exponential deconvolved out of it -- what is left is the
        #: response, and it goes into the model exactly as the water one does
        #: AT THE MAGIC ANGLE, because a free dye is not a single exponential in
        #: a polarised channel: its rotational depolarisation (a few hundred
        #: picoseconds for Rh110 in water) puts a rise into VH and a drop into
        #: VV, and Rh110's two channels here rise in 0.58 and 0.90 ns where the
        #: water response takes 0.70 -- deconvolving either alone by the 4 ns
        #: exponential returned a one-channel spike. (1 - 3 l2) VV +
        #: (2 - 3 l1) g VH is the isotropic decay, IRF * exp(-t/tau), and the
        #: one response shape it yields serves both green detectors; their own
        #: shift and background stay per-detector nuisances. That the two
        #: detectors' water responses differ by ten per cent in width is a
        #: cost of this choice, stated here rather than hidden.
        #: THE WHOLE PERIOD, NOT A WINDOW.  A 4 ns dye at 32 MHz has not decayed
        #: when the next pulse comes (e^-7.8 of its peak wraps round), and the
        #: red pulse does not excite it, so its histogram is one periodic decay
        #: from end to end: 5,700 counts per channel at 13-16 ns, 875 at
        #: 19-27 ns, 295 at 28-31 ns. `reference_dye` cuts it at +14 ns for the
        #: delta-function route, and that cliff is what Richardson-Lucy turned
        #: into a one-channel spike (measured: the windowed analytic case
        #: recovers the response to 62 % of its peak, the full period to 5 %).
        #: The background is what sits just BEFORE the pulse, where the
        #: wrapped tail is 4e-4 of the peak.
        _, info_ref = reference_dye(d, which='rhd110', n=n)
        g_ref = 1.0 / cal['g_green']; l1, l2 = cal['l1'], cal['l2']
        by_colour = {}
        for colour in sorted({DETECTORS[det][0] for det in dets}):
            vv_det = next(k for k, v in DETECTORS.items() if v == (colour, 'parallel'))
            vh_det = next(k for k, v in DETECTORS.items() if v == (colour, 'perpendicular'))
            vv = np.asarray(d['reference'][('rhd110', colour, 'parallel')], float)[:n]
            vh = np.asarray(d['reference'][('rhd110', colour, 'perpendicular')], float)[:n]
            pv, ph = int(np.argmax(vv)), int(np.argmax(vh))
            vv = np.maximum(vv - np.median(vv[:max(pv - 15, 1)]), 0.0)
            vh = np.maximum(vh - np.median(vh[:max(ph - 15, 1)]), 0.0)
            #: the two channels' pulses sit at different channels; align the
            #: perpendicular one to the parallel one before combining
            vh = np.roll(vh, pv - ph)
            ma = (1.0 - 3.0 * l2) * vv + (2.0 - 3.0 * l1) * g_ref * vh
            #: the water response, its flat background removed, is the start;
            #: it sits where the response is, so no window is imposed on the
            #: result beyond the water response's own support
            w0 = np.asarray(irf_h20[vv_det], float)
            w0 = np.where(w0 > 0, np.maximum(w0 - info[vv_det]['flat_per_channel'], 0.0), 0.0)
            f = rl_deconvolve(ma, TAU_RH110_NS, cal['dt'], n_iter=rl_iterations, f0=w0)
            f = np.where(w0 > 0, f, 0.0)
            by_colour[colour] = (f * (ma.sum() / max(f.sum(), 1e-300)), info_ref[vv_det])
        for det in dets:
            f, inf = by_colour[DETECTORS[det][0]]
            irf[det] = f.copy()
            info[det] = dict(inf, source=f"Rh110 magic angle, {rl_iterations} Richardson-Lucy iterations at {TAU_RH110_NS} ns",
                             background_fraction=0.01)
        ref_dets = ()          # a measured response now, not a reference-dye one
    elif irf_kind not in ('h20', 'analytic'):
        raise ValueError(f"irf must be 'h20', 'rh110' or 'analytic', not {irf_kind!r}")
    keys, pairs = [], {}
    for samp in samples:
        for det in dets:
            k = (samp, f'{det}_{KIND[det]}')
            kp = (samp, f'{det[0]}2{det[1:]}_{KIND[det]}')
            keys += [k, kp]; pairs[k] = kp
    E['pie_pairs'] = pairs

    y_np = {k: v for k, v in y_np.items() if k in pairs}
    mask_np = {k: v for k, v in mask_np.items() if k in pairs}
    y_full = {k: L.tt(v) for k, v in y_np.items()}
    if rebin:
        #: the counts in the widening bins; a bin is masked only if every
        #: channel in it is
        y = {k: E['R'] @ y_full[k] for k in y_np}
        masks = {k: ((E['R'] @ L.tt(mask_np[k])) > 0).to(torch.float64) for k in y_np}
    else:
        y = y_full
        masks = {k: L.tt(mask_np[k]) for k in y_np}
    #: the background of each histogram as a FRACTION of its counts, from the
    #: flat parts of the decay -- a starting point for a node, not a constant
    bkg_med, tot = {}, {}
    for k, v in y_np.items():
        b = background(v, cal)
        tot[k] = float(v.sum())
        bkg_med[k] = max(b['per_channel'] * n / max(tot[k], 1.0), 1e-6)
    ref = {samp: next(k for k in pairs if k[0] == samp) for samp in samples}
    scale_med = {samp: max(tot[ref[samp]], 1.0) for samp in samples}
    irf_bg_med = {det: max(min(info[det]['background_fraction'], 0.9), 1e-3) for det in dets}

    analytic = (irf_kind == 'analytic')
    if analytic:
        #: the water response's peak and width set the priors of the skewed
        #: Gaussian; `analytic_irf` centres it at 0.10 T + shift
        E.setdefault('T', E['period'])
        shift0, width0 = {}, {}
        for det in dets:
            r = np.asarray(irf_h20[det], float); pk = int(np.argmax(r)); half = r[pk] / 2.0
            lo = pk
            while lo > 0 and r[lo] > half:
                lo -= 1
            hi = pk
            while hi < len(r) - 1 and r[hi] > half:
                hi += 1
            width0[det] = max((hi - lo) * cal['dt'] / 2.355, 0.5 * cal['dt'])
            shift0[det] = pk * cal['dt'] - 0.10 * E['T']
        w0 = float(np.mean(list(width0.values())))
        E['instrument'] = ('skewed Gaussian, priors from the water response', w0, 0.0, 0.0)
    V = L.default_variables(E, keys, n_coef=n_coef, scale_medians=scale_med,
                            irf_shape=analytic, bkg_medians=bkg_med, ref_dets=ref_dets,
                            irf_bg_medians=(None if analytic else irf_bg_med))
    for v in V:
        if v.name.startswith('irf_tauref_'):
            #: the dye's lifetime is known to a few per cent; the node is there
            #: so that the fit can say whether it agrees
            v.prior = L.LogNormal(TAU_RH110_NS, 0.05 * L.LN10)
        if analytic and v.name.startswith('irf_shift_'):
            det = v.name[len('irf_shift_'):]
            v.prior = L.Gaussian(shift0[det], 0.05)
    #: THE BACKGROUND IS MEASURED, SO IT GETS A MEASURED PRIOR.
    #:
    #: The prototype gives every background half a decade, which is right when
    #: the only thing known about it is that it is small. Here it is not: the
    #: flat parts of each histogram give it directly, and notebook 19 quotes two
    #: windows separately so that a disagreement between them shows.
    #:
    #: Leaving it half a decade wide let the start put 48 counts per channel into
    #: the donor-only parallel histogram where the data support about 15, and
    #: that single number was most of the misfit: with the flat component removed
    #: the two polarisations agree with each other (tails 9.6 and 10.7) and the
    #: model sits within a factor of 1.7 of the data, and with it the parallel
    #: channel is five times the perpendicular where the data say 1.2.
    for v in V:
        if v.name.startswith('bkg_'):
            k = tuple(v.name[len('bkg_'):].split('_', 1))
            if k in bkg_med:
                v.prior = L.LogNormal(bkg_med[k], 0.15 * L.LN10)
    return_bkg_sd = 0.15
    inst = L.InstrumentModel(E, 'analytic') if analytic else \
        L.InstrumentModel(E, 'measured', {det: L.tt(irf[det]) for det in dets})
    ps = L.PSplineFactor(n_coef, spl=spl)
    g = L.FactorGraph(E, keys, V, L.PoissonCountsFactor({k: y[k] for k in pairs}, mask=masks),
                      inst, spl, ps)
    g.rel = rel
    g.start_y = {k: y[k] * E['win_green'] for k in pairs}
    return dict(L=L, E=E, Ep=E, rel=rel, spl=spl, keys=keys, pairs=pairs, graph=g,
                y=y, y_full=y_full, t=t_axis, masks=masks, irf=irf, irf_info=info, offset_channels=off,
                irf_kind=irf_kind,
                irf_h20=irf_h20,
                cal=cal, loaded=d, n=n, n_coef=n_coef,
                bkg_medians=bkg_med, scale_medians=scale_med, irf_bg_medians=irf_bg_med)


def fit(m, lam_nodes=(1.0, 0.0, -1.0), seed=0, verbose=True, accelerate=False, fixed=None):
    """The Laplace posterior of the whole model on the twelve histograms.

    **Automatic differentiation, not the analytic Jacobian.** The hand-written
    amplitude Jacobian covers three scopes and this measurement has six -- every
    sample is seen under both pulses -- and the prototype raises rather than
    quietly returning the wrong derivative. That is the right behaviour and it
    is why this fit is minutes rather than seconds.
    """
    import torch
    L = m['L']
    #: THE ANALYTIC JACOBIAN, now that it covers the six scopes (R2 of
    #: okf/prd-real-data-fast.md): 16 ms against 1.78 s per scoring iteration,
    #: gated against forward-mode AD at 6e-16 and sharing the AD path's modes.
    L.Laplace.analytic = True
    #: START FROM THE MEASURED BACKGROUND.  `start_from_data` fits it by a
    #: non-negative solve on a 33-column basis whose long-lifetime members are
    #: nearly identical, and on this measurement it returned 0.0714 and 0.0000
    #: for two channels that must agree -- 48 counts per channel into a
    #: histogram whose flat parts say 15.  Notebook 19 measures it directly, so
    #: the start uses that and the fit refines it under a prior a sixth of a
    #: decade wide.  Worth 18.6 to 13.0 in the deviance at the start, and the
    #: difference between a fit that converges and one that does not.
    th0, _ = L.start_from_data(m['graph'], m['y'], verbose=False, method='mem')
    for k, med in m['bkg_medians'].items():
        nm = f'bkg_{k[0]}_{k[1]}'
        if nm in m['graph'].offsets:
            a, b = m['graph'].offsets[nm]
            th0[a:b] = m['graph'].index[nm].transform.to_unconstrained(L.tt([med]))
    m['theta_start'] = th0
    if accelerate:
        #: OFF AGAIN.  The pointwise gate (`fast_forward.gate`) passes at 1e-15,
        #: and the WHOLE-FIT gate fails: the same fit -- g held at 0.9328, one
        #: penalty node -- reaches D/dof 2.279 and converges on the prototype
        #: path, 13.2 and does not converge on this one (2026-09-14). A gate at
        #: points in parameter space says nothing about the scoring path, which
        #: goes through `graph.inst.basis` and the Jacobian rather than through
        #: `log_posterior`. Stays off until that gate passes too.
        #:
        #: What the pointwise gate DID establish, and is kept:
        #:
        #: It was off, because the accelerated log posterior differed from the
        #: prototype's by 590,000 nats and converged to a deviance per degree of
        #: freedom of 8841 where the prototype reached 1.313 -- while the
        #: forward model it is built on agreed at 5e-16. Checking the piece I
        #: changed could not catch a CONSUMER of it. Two defects were hiding
        #: there, and both were invisible to the simulated benchmark:
        #:
        #:   * `stage2` keyed the scope dictionary by SAMPLE. Without
        #:     interleaved excitation sample and scope are the same word; with
        #:     it, each sample appears under both pulses, and the red-pulse
        #:     partner of every histogram was handed the GREEN pulse's
        #:     amplitudes -- a factor of 500 on the donor-only sample.
        #:   * the shift was a phase ramp on the kernel, while the prototype
        #:     shifts the response with `shift_irf_fft`, which CLAMPS the ramp's
        #:     ringing before renormalising. Agreement therefore held only where
        #:     every shift sat at zero.
        #:
        #: `fast_forward.gate` now compares the two log posteriors under a
        #: perturbation of each parameter group in turn; it reports 1e-15 on
        #: this measurement and on the analytic eight-decay configuration of
        #: S5 (run S5ord3, realisation 134), and it would have caught both.
        import fast_forward as FF
        FF.accelerate(m, m['graph'])
    gen = torch.Generator().manual_seed(int(seed))
    #: the penalty grid, node by node, from that start -- rather than
    #: `fit_sample`, which computes its own
    g = m['graph']
    tr = g.index['log10_lam'].transform
    #: HOLD A CALIBRATION CONSTANT AT A MEASURED VALUE.  `g` and `r0_d` are two
    #: handles on the same VV/VH ratio, and with both free no penalty node
    #: converges on this measurement -- the optimiser walks between them.  The
    #: reference dye measures `g` directly (Rh110 in water is depolarised within
    #: a few hundred picoseconds, so VV/VH is the detection ratio and nothing
    #: else), so pinning it there is using a measurement, not fixing a fit.
    held = {k: g.index[k].transform.to_unconstrained(L.tt([float(v)]))
            for k, v in (fixed or {}).items()}
    nodes, th_prev = {}, th0
    for lg in lam_nodes:
        gi = g.with_fixed(log10_lam=tr.to_unconstrained(L.tt([float(lg)])), **held)
        r = L.laplace_at(gi, gi.restrict(gi, th_prev), optimiser='fisher',
                         verbose=verbose, hessian='fisher')
        r['graph'] = gi; r['log10_lam'] = float(lg)
        nodes[float(lg)] = r
        if r.get('converged'):
            th_prev = th0.clone()
            th_prev[:] = th0
            for n_, (a, b) in gi.offsets.items():
                if n_ in g.offsets:
                    a0, b0 = g.offsets[n_]
                    th_prev[a0:b0] = r['theta'][a:b]
    #: a node that did not converge has no evidence, and mixing over it would
    #: put a NaN through everything downstream
    ev = np.array([float(nodes[float(l)].get('evidence', np.nan)) for l in lam_nodes], float)
    conv = np.array([bool(nodes[float(l)].get('converged')) for l in lam_nodes])
    #: NOT CONVERGING IS A RESULT, AND IT GETS REPORTED RATHER THAN RAISED.
    #: On this measurement no node converges while `g` is free: the g factor and
    #: the donor's fundamental anisotropy are two handles on the same VV/VH
    #: ratio, and the optimiser walks between them.  Raising here hid that
    #: behind a traceback and left the notebook with no outputs at all; Rule 0
    #: excludes and COUNTS a failing fit, which needs the fit in hand.
    converged_any = bool(conv.any())
    if not converged_any:
        conv = np.ones_like(conv)
    ok = np.isfinite(ev) & conv
    if not ok.any():
        #: the nodes reached their modes but the curvature there is not positive
        #: definite, so there is no Laplace evidence to mix with.  That is a
        #: statement about the posterior -- some direction is flat or worse --
        #: and it is reported rather than raised, because the modes are still
        #: the modes and a reader needs to see them.
        ok = conv
        evidence_available = False
    else:
        evidence_available = True
    w = np.zeros_like(ev)
    if evidence_available:
        w[ok] = np.exp(ev[ok] - ev[ok].max())
    else:
        w[ok] = 1.0
    w = w / w.sum()
    best = float(np.asarray(lam_nodes, float)[int(np.argmax(w))])
    post = dict(nodes[best])
    post.update(nodes=nodes, lam_nodes=[float(x) for x in lam_nodes], weights=w,
                ev=ev, best_lam=best, evidence_available=bool(evidence_available),
                converged_any=converged_any, fixed=dict(fixed or {}))
    vals, _ = post['graph'].unpack(post['theta'])
    lam = post['graph'].expected_counts(vals)
    rows, dev, dof = L.rule0(post['graph'], m['y'], lam)
    post.update(lam={k: v.detach() for k, v in lam.items()}, rows=rows, dev=dev, dof=dof,
                p=post['graph'].distribution(vals).detach().numpy(),
                converged=[bool(nodes[float(l)].get('converged')) for l in lam_nodes])
    return post


def rule0(m, post):
    """Rule 0 per histogram: the Poisson deviance per degree of freedom and a
    runs test on the weighted residuals, with the deviance compared against a
    reference MEASURED by drawing Poisson data at the fitted means."""
    from bd import P
    rows = [dict(channel=f'{k[0]} {k[1]}', counts=float(m['y'][k].sum()),
                 dpd=r['dpd'], runs_p=r['runs_p']) for k, r in post['rows'].items()]
    ref = P.poisson_reference(post, n_draw=150, seed=0)
    dpd = post['dev'] / post['dof']
    rows.append(dict(channel='all', counts=float(sum(float(m['y'][k].sum()) for k in m['pairs'])),
                     dpd=dpd, runs_p=float('nan')))
    return rows, ref, float((dpd - ref[0]) / max(ref[1], 1e-12))


# --------------------------------------------------------------------------
# every nuisance, against its prior
# --------------------------------------------------------------------------

#: the three that are not nuisances: the distance distribution's coefficients,
#: the lifetime spectrum, and the roughness weight that is integrated out
NOT_NUISANCE = ('c', 'spec_eps', 'log10_lam')


def nuisances(m, post, n_draw=4000, seed=0, exclude=NOT_NUISANCE):
    """Each nuisance's posterior beside its prior, in the units it is quoted in.

    The Laplace approximation gives the posterior covariance in the coordinate
    the fit works in; what a reader wants is the constrained value -- a g factor,
    a fraction, a shift in nanoseconds -- so the posterior sd is carried through
    the transform by the delta method and the prior is characterised by drawing
    from it.

    **The column to read is the last one.** `1 - (posterior sd / prior sd)^2` is
    the fraction of the prior's variance the data removed. Near one the number
    is a measurement; near zero it is the prior, whatever the fit prints, and on
    a real sample there is no truth to notice the difference.
    """
    import torch
    L = m['L']
    g = post['graph']
    Sig = post['Sigma']; th = post['theta']
    gen = torch.Generator().manual_seed(int(seed))
    rows = []
    for v in g.free:
        if v.name in exclude:
            continue
        a, b = g.offsets[v.name]
        #: the prior in the constrained coordinate, by drawing from it
        draws = torch.stack([v.transform.to_constrained(v.sample_z(gen).reshape(-1))
                             for _ in range(n_draw)])
        pri_mu = draws.mean(0); pri_sd = draws.std(0)
        z = th[a:b].detach().clone()
        x0 = v.transform.to_constrained(z)
        for i in range(v.size):
            #: the delta method through this variable's own transform
            h = 1e-5 * max(abs(float(z[i])), 1.0)
            zp = z.clone(); zp[i] += h
            dxdz = float((v.transform.to_constrained(zp) - x0)[min(i, x0.numel() - 1)] / h)
            #: a fit that did not converge carries no covariance: the mode is
            #: reported with no width rather than not at all
            sd_z = float(Sig[a + i, a + i]) ** 0.5 if Sig is not None else float('nan')
            sd_x = abs(dxdz) * sd_z
            j = min(i, pri_sd.numel() - 1)
            ps = float(pri_sd[j])
            rows.append(dict(
                name=v.name if v.size == 1 else f'{v.name}[{i}]',
                group=v.group, doc=v.doc,
                posterior=float(x0[min(i, x0.numel() - 1)]), posterior_sd=sd_x,
                prior=float(pri_mu[j]), prior_sd=ps,
                learned=float(max(0.0, 1.0 - (sd_x / ps) ** 2)) if ps > 0 else float('nan')))
    return rows


def plot_nuisances(rows, groups=('phys', 'cal', 'inst'), ax=None, max_per_group=None):
    """Every nuisance drawn against its prior: the prior interval in grey, the
    posterior on top of it, each normalised to its own prior so that a g factor
    and a background fraction can sit on one axis.

    A bar that fills its grey band is a parameter the measurement did not
    determine. A bar much narrower than the band, and displaced from it, is one
    the measurement moved.
    """
    import matplotlib.pyplot as plt
    sel = [r for r in rows if r['group'] in groups and r['prior_sd'] > 0]
    sel.sort(key=lambda r: (groups.index(r['group']), -r['learned']))
    if max_per_group:
        keep, seen = [], {}
        for r in sel:
            seen[r['group']] = seen.get(r['group'], 0) + 1
            if seen[r['group']] <= max_per_group:
                keep.append(r)
        sel = keep
    n = len(sel)
    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, max(3.0, 0.24 * n)))
    colour = {'phys': 'C0', 'cal': 'C2', 'inst': 'C1'}
    for i, r in enumerate(sel):
        z = (r['posterior'] - r['prior']) / r['prior_sd']
        w = r['posterior_sd'] / r['prior_sd']
        ax.barh(i, 4.0, left=-2.0, height=0.75, color='0.88', zorder=1)
        ax.barh(i, 4.0 * w, left=z - 2.0 * w, height=0.5,
                color=colour.get(r['group'], 'C4'), zorder=2)
        ax.plot([z], [i], '|', color='k', ms=6, zorder=3)
    ax.axvline(0.0, color='0.4', lw=0.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f'{r["name"]}  ({r["learned"]:.2f})' for r in sel], fontsize=7)
    ax.set_xlabel('posterior, in prior standard deviations from the prior mean'
                  '  (grey: the prior, +-2 sd)')
    ax.set_xlim(-3.2, 3.2); ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour[k]) for k in colour if any(r['group'] == k for r in sel)]
    labels = [{'phys': 'physics', 'cal': 'calibration', 'inst': 'instrument'}[k]
              for k in colour if any(r['group'] == k for r in sel)]
    ax.legend(handles, labels, fontsize=7, loc='lower right')
    return ax


# --------------------------------------------------------------------------
# the reference dye
# --------------------------------------------------------------------------

def g_from_reference(loaded, which='rhd110', windows=((10, 40), (40, 90), (90, 150), (150, 220))):
    """The `g` factor measured from the reference dye -- and the convention
    question it settles.

    Rhodamine 110 in water rotates in a fraction of a nanosecond, so a few
    hundred picoseconds after the pulse its emission is depolarised and the two
    polarised channels differ only by their detection efficiencies. In the
    model's convention the perpendicular channel is DIVIDED by `g`, so for a
    depolarised emitter `g = VV / VH`. That is what this measures, after
    subtracting each channel's own background.

    **Paris quotes the reciprocal.** The measurement is 0.93 to 0.96 in every
    window of both repeats and both colours, while the calibration file states
    1.0720 and 1.1056 -- whose reciprocals are 0.9328 and 0.9045. All eight
    measurements sit at the reciprocal (tpeulen, 2026-09-14: "Paris may have
    some funky inverse g factor"), and that also settles which half of each file
    is which: the first is the parallel channel, as the water measurement's
    twofold excess already said.

    The red detectors come out four to five per cent above their reciprocal,
    which is the direction a residual anisotropy pushes `VV/VH`, so the two
    disagreements that looked contradictory are one convention and one small
    physical effect.
    """
    cal = loaded['cal']
    out = {}
    for colour, stated in (('green', cal['g_green']), ('red', cal['g_red'])):
        p = np.asarray(loaded['reference'][(which, colour, 'parallel')], float)
        sper = np.asarray(loaded['reference'][(which, colour, 'perpendicular')], float)
        pk = int(np.argmax(p[:255]))
        bp, bs = float(np.median(p[430:481])), float(np.median(sper[430:481]))
        ratios = [float((p[pk + a:pk + b] - bp).sum() / max((sper[pk + a:pk + b] - bs).sum(), 1.0))
                  for a, b in windows]
        mu = float(np.mean(ratios))
        out[colour] = dict(peak_channel=pk, ratios=ratios, measured=mu,
                           stated=float(stated), reciprocal=float(1.0 / stated),
                           matches='the reciprocal' if abs(mu - 1.0 / stated) < abs(mu - stated)
                                   else 'the stated value',
                           windows=[(pk + a, pk + b) for a, b in windows])
    return out


def reference_dye(loaded, which='rhd110', pad=(15, 220), n=None) -> dict:
    """The reference dye's decay, per detector, as a RESPONSE.

    A response measured on water is scattered laser light and nothing else,
    which is what makes it the right shape -- and on this instrument it is also
    weak, seventy per cent pedestal in the red detectors. A reference dye is
    bright instead, and the model can use it directly: if the response is a
    dye's decay `R = IRF * exp(-t/tau_ref)` rather than the instrument's own,
    then

        IRF * e_i = R + (1/tau_ref - 1/tau_i) (R * e_i)

    exactly, with no deconvolution anywhere (Zuker, Szabo, Bramall, Krajcarski &
    Selinger, Rev. Sci. Instrum. 56, 14, 1985). `basis_from_irf` implements it
    and `irf_tauref_<detector>` is the node that carries the dye's lifetime, so
    using this needs no new mathematics -- only the decay and a prior on it.

    Rhodamine 110 is about 4 ns in water, which is long; the method wants a
    reference much shorter than the lifetimes being measured, so this is offered
    and its cost has to be measured rather than assumed.
    """
    n = int(loaded['cal']['n_channels']) if n is None else int(n)
    out, info = {}, {}
    for det, (colour, pol) in DETECTORS.items():
        h = np.asarray(loaded['reference'][(which, colour, pol)], float)
        peak = int(np.argmax(h[:255]))
        a, b = max(peak - pad[0], 0), min(peak + pad[1], len(h))
        seg = np.zeros(len(h)); seg[a:b] = h[a:b]
        out[det] = seg[:n].copy()
        flat = float(np.median(h[430:481]))
        info[det] = dict(peak_channel=peak, counts=float(seg.sum()), peak=float(seg.max()),
                         flat_per_channel=flat,
                         background_fraction=float(min(flat * (b - a) / max(seg.sum(), 1), 0.95)))
    return out, info


def apply_calibration_priors(m, g_sd=0.02, l_sd=0.005, verbose=True):
    """Replace the prototype's simulated calibration priors with THIS
    spectrometer's.

    The model's defaults are the simulation's -- `g` 1.15, `l1` 0.03, `l2` 0.02 --
    and this instrument is 1.0720, 0.0175 and 0.0526. **`l2` is 3.3 of the
    default prior's standard deviations away**, and it controls how much
    parallel light enters the perpendicular channel, so leaving it there pulls
    the anisotropy and with it everything the perpendicular histograms say.

    The widths are the point of judgement. `g` and the polarisation mixing are
    calibrations somebody measured, not guesses, so they are given narrow priors
    -- but not fixed, because a calibration measured on another day is a prior
    and not a fact.
    """
    L = m['L']; cal = m['cal']
    #: THE RECIPROCAL. Paris quotes a g the model would call 1/g, which
    #: `g_from_reference` settles against the reference dye: the measured VV/VH
    #: of a depolarised emitter is 0.93 to 0.96, and the file says 1.07 and
    #: 1.11. Using the file's number as written would have divided the
    #: perpendicular channel by 1.07 where it should be divided by 0.93 -- a
    #: 15 % error in the anisotropy, in the wrong direction, on every
    #: perpendicular histogram.
    want = {'g': (1.0 / cal['g_green'], g_sd), 'g_r': (1.0 / cal['g_red'], g_sd),
            'l1': (cal['l1'], l_sd), 'l2': (cal['l2'], l_sd),
            'QY_D': (cal['qy_donor'], 0.10), 'QY_A': (cal['qy_acceptor'], 0.10),
            'EX_AG': (cal['direct_excitation'], 0.30),
            'C_RD': (cal['crosstalk'], 0.20)}
    changed = []
    for v in m['graph'].V:
        if v.name not in want:
            continue
        mu, sd = want[v.name]
        old = v.prior
        if isinstance(old, L.Gaussian):
            v.prior = L.Gaussian(mu, sd)
        elif isinstance(old, L.LogNormal):
            v.prior = L.LogNormal(mu, sd)
        else:
            continue
        changed.append((v.name, mu, sd))
    if verbose:
        print(f'{"parameter":<8}{"prior median":>14}{"prior sd":>11}   from')
        for n_, mu, sd in changed:
            src = ('the calibration file, INVERTED' if n_ in ('g', 'g_r')
                   else 'the calibration file' if n_ in ('l1', 'l2')
                   else "Alex's single-molecule fit")
            print(f'{n_:<8}{mu:>14.4f}{sd:>11.4f}   {src}')
    return changed
