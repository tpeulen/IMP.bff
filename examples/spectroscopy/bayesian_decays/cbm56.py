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

import math
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
    It runs from 15 channels before the peak to 150 after (10.8 ns). The raw
    histogram is the full 512 channels and flat at 80-90 counts per channel
    outside its pulse (tpeulen, prompt 415: "why is the water meas half the
    length"), so the window loses nothing but flat channels -- and widening
    it to the red pulse (`pad=(15, 215)`) was MEASURED to make the fit worse,
    not better: the donor-only fit that passes at 1.219 under this window
    reaches 1.367 and does not converge under the wide one, with the
    perpendicular detector's background fraction walking to 0.44. More flat
    channels make `irf_bg` less identifiable against the flat column, and the
    walk takes it. The window stays; the wide one is a parameter.
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


def histograms(loaded, n=None, mask_edges=None) -> dict:
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
        #: REF: the reference dye as a sample of the global analysis (prompt 419)
        h = np.asarray(loaded['reference'][('rhd110', colour, pol)], float)[:n]
        y[('REF', f'{det}_{KIND[det]}')] = h
        mask[('REF', f'{det}_{KIND[det]}')] = np.ones(n)
    if mask_edges is not None:
        #: A6: the period's edge channels -- channel 0 is the TAC's edge (5-6
        #: channels hold half the pre-rise flat), and 488 x 64 ps = 31.232 ns
        #: against a 31.249 ns period leaves the last channel partial
        a_, b_ = mask_edges
        for k in mask:
            if a_:
                mask[k][:a_] = 0.0
            if b_:
                mask[k][n - b_:] = 0.0
    return y, mask


# --------------------------------------------------------------------------
# the model's environment, on this spectrometer's time axis
# --------------------------------------------------------------------------

#: THE MAPS' CONSTRUCTION (okf/prd-real-data-fast.md A2, 2026-09-14): 'exact'
#: builds every periodic column with the exact bin-integrated kernel and the
#: rotational and FRET maps with targets built the basis's own way, projected by
#: ridge -- 1.9e-5 of the decay peak against the exact column, where 'legacy'
#: (the trapezoid kernel, sampled targets, NNLS: every fit before 18:30 that
#: day) is 3.6e-2.  The acceptor maps keep their own construction; they enter
#: the green detectors only through C_GA and are the red phase's to gate.
MAPS = 'exact'


def environment(n_coef=25, n=488, cache=True, verbose=True, rho_grid=None, maps=None):
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

    import s53_phase1_pseudolik as S53
    maps = MAPS if maps is None else maps
    if maps == 'exact':
        #: the ridge by QR (PRD-143 A3, 2026-09-15): the normal equations carried 3e-7 of
        #: error in the map coefficients; the cache tag changes with it
        S53.KERNEL, S.MAP_TARGET, S.MAP_SOLVER = 'exact', 'periodic', 'ridgeqr'
    elif maps == 'exact_normal_equations':
        #: the maps as built before 2026-09-15 (the v17 record was fitted on these)
        S53.KERNEL, S.MAP_TARGET, S.MAP_SOLVER = 'exact', 'periodic', 'ridge'
    elif maps == 'legacy':
        S53.KERNEL, S.MAP_TARGET, S.MAP_SOLVER = 'trapezoid', 'sampled', 'nnls'
    else:
        raise ValueError(f"maps must be 'exact', 'exact_normal_equations' or 'legacy', not {maps!r}")
    mtag = '' if maps == 'legacy' else f'_{S53.KERNEL}_{S.MAP_TARGET}_{S.MAP_SOLVER}'
    cal = CAL or _cal()
    dt = cal['dt']
    rel, edges = R3.grid(M.N_REL)
    S.TAU_REF = L.TAU_0
    S.R_GRID = S.R0_FOERSTER * rel
    ck = d / 'ckpt' / 'homog'
    ck.mkdir(parents=True, exist_ok=True)
    f = ck / f'cbm56_env_{n}_{M.N_REL}{mtag}.pt'
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
    #: THE ROTATIONAL GRID (R1(a), prompt 416: "residuals are awful!"): the
    #: prototype's runs 0.1-20 ns in nine points, and under a tail-free
    #: response the donor-only parallel channel misfits early while the
    #: perpendicular one passes -- a polarised, early defect, which is what a
    #: rotational component outside the grid looks like. `rho_grid` widens it;
    #: the prior on its weights (`default_variables`) reads the same module
    #: constant, so both follow.
    if rho_grid is not None:
        M.RHO_GRID = np.asarray(rho_grid, float)
    tag = f'{len(M.RHO_GRID)}_{M.RHO_GRID.min():g}_{M.RHO_GRID.max():g}'
    f2 = ck / f'cbm56_rho_{n}_{tag}{mtag}.pt'
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
    f3 = ck / f'cbm56_rhoa_{n}_{len(M.RHO_A_GRID)}{mtag}.pt'
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

#: THE REFERENCE DYES' PRIORS, declared and cited (tpeulen, 2026-09-14: "for
#: reference dyes, like Rh110, you could/should define a prior").  A field is a
#: (value, sd, unit, source) tuple or None; None means no citable value is to
#: hand and the model's generic prior stands -- deliberately, rather than a
#: number that looks like a measurement.  `apply_reference_priors` applies it.
REFERENCE_DYES = {
    'rhd110': dict(
        name='Rhodamine 110 in water',
        tau=(4.00, 0.05, 'log10 decades', 'Magde, Rojas & Seybold, Photochem. Photobiol. 75, 327 (2002)'),
        rho=(0.10, 0.20, 'ns, range', 'tpeulen, 2026-09-14 (prompt 425)'),
        r0=None,           # fundamental anisotropy: no citable value supplied
    ),
}


def apply_reference_priors(m, dye='rhd110', verbose=True):
    """Set the REF sample's priors from `REFERENCE_DYES[dye]`: the lifetime
    (`log10_tau_ref`, Gaussian in log10 ns), and -- where the table carries a
    value -- the fundamental anisotropy (`r0_ref`, a Gaussian on its bounded
    value is not available, so a narrowed uniform window of +-2 sd) and the
    rotational time (a LogisticNormal bump on `w_rho_ref`)."""
    L = m['L']; d = REFERENCE_DYES[dye]; done = []
    for v in m['graph'].V:
        if v.name == 'log10_tau_ref' and d.get('tau'):
            mu, sd = d['tau'][0], d['tau'][1]
            v.prior = L.Gaussian(math.log10(mu), sd); done.append(('log10_tau_ref', f'{mu} ns +- {sd} decades', d['tau'][3]))
        if v.name == 'w_rho_ref' and d.get('rho'):
            #: a range (lo, hi): a log-normal bump at the geometric centre with
            #: half the range in decades as its width, as the donor's own
            #: rotational prior is built (`log_bump`, `alr_of`, unit sd on ALR)
            lo, hi = d['rho'][0], d['rho'][1]
            centre = math.sqrt(lo * hi); width = 0.5 * math.log10(hi / lo)
            grid = np.asarray(m['E']['rho'], float)
            v.prior = L.LogisticNormal(L.alr_of(L.log_bump(grid, centre, width)), np.ones(len(grid) - 1))
            done.append(('w_rho_ref', f'{lo}-{hi} ns (bump at {centre:.3f} ns, {width:.2f} decades)', d['rho'][3]))
        if v.name == 'r0_ref' and d.get('r0'):
            mu, sd = d['r0'][0], d['r0'][1]
            lo, hi = max(0.0, mu - 2 * sd), min(0.4, mu + 2 * sd)
            v.transform = L.Logit(lo, hi); v.prior = L.Uniform(lo, hi); done.append(('r0_ref', f'[{lo:.3f}, {hi:.3f}]', d['r0'][3]))
    if verbose:
        for n_, what, src in done:
            print(f'  reference prior {n_}: {what}  ({src})')
        for f_ in ('rho', 'r0'):
            if not d.get(f_):
                print(f'  reference prior {f_}: none declared -- the generic prior stands')
    return done


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


def generalized_normal(t, loc, scale, shape):
    """chisurf's response shape (`generalized_normal_distribution`, forwarded to
    IMP.bff `Distributions.h`): Hosking's generalized normal, a Gaussian for
    shape 0 and a shifted, possibly reversed log-normal otherwise, so that a
    detector's transit-time asymmetry has one parameter. Unit sum over `t`."""
    u = (t - loc) / scale
    if abs(shape) < 1e-12:
        y = u; jac = np.ones_like(u)
    else:
        arg = 1.0 - shape * u
        ok = arg > 0
        y = np.where(ok, -np.log(np.where(ok, arg, 1.0)) / shape, np.inf)
        jac = np.where(ok, 1.0 / np.where(ok, arg, 1.0), 0.0)
    f = np.where(np.isfinite(y), np.exp(-0.5 * y * y) * jac, 0.0)
    return f / max(f.sum(), 1e-300)


def emg_pulse(t, loc, sigma, tau):
    """An exponentially modified Gaussian: Gaussian timing jitter of width
    `sigma` convolved with an exponential transit tail of time `tau` -- the
    textbook single-photon detector response (Grushka, Anal. Chem. 44, 1733,
    1972, for the closed form). Unit sum over `t`. Where the skewed Gaussian
    could not follow a 0.26 ns rise against a 0.7 ns width, this can."""
    from scipy.special import erfcx
    tau = max(tau, 1e-6); sigma = max(sigma, 1e-6)
    z = (sigma / tau - (t - loc) / sigma) / np.sqrt(2.0)
    #: erfcx keeps the product finite where the plain exp overflows; far in
    #: the Gaussian's own tail it still overflows harmlessly and is zeroed
    with np.errstate(over='ignore', invalid='ignore'):
        f = 0.5 / tau * np.exp(-0.5 * ((t - loc) / sigma) ** 2) * erfcx(z)
    f = np.where(np.isfinite(f), f, 0.0)
    return f / max(f.sum(), 1e-300)


def peak_fit_response(h, dt, stop_after_peak=1.0, start_fraction=0.05, pre=8, pulse='gn'):
    """chisurf's way with a contaminated water measurement (tpeulen, prompt
    414): a skewed Gaussian fitted to the PEAK REGION only -- from where the
    rise passes `start_fraction` of the peak to `stop_after_peak` ns after it --
    where scattered laser light dominates and the dirt's fluorescence has not
    yet had time to matter; the tail is then the fitted shape's own, not the
    dirt's. Poisson maximum likelihood over (loc, scale, shape, amplitude,
    flat) by scipy.optimize.minimize (Nelder-Mead from a moment start, then
    BFGS), reported with the deviance per degree of freedom of the region so
    that a shape the region rejects is visible."""
    from scipy.optimize import minimize
    h = np.asarray(h, float); n = len(h); t = np.arange(n) * dt
    pk = int(np.argmax(h)); peak = h[pk]
    a = pk
    while a > 0 and h[a] > start_fraction * peak:
        a -= 1
    a = max(a - pre, 0)
    b = min(pk + int(round(stop_after_peak / dt)) + 1, n)
    tw, hw = t[a:b], h[a:b]
    flat0 = max(float(np.median(h[max(a - 40, 0):a])) if a > 0 else 0.0, 1e-3)
    def model_(q):
        loc, lsc, shape, lamp, lflat = q
        if pulse == 'emg':
            #: (loc, log sigma, log tau, log amplitude, log flat)
            return np.exp(lamp) * emg_pulse(tw, loc, np.exp(lsc), np.exp(shape)) + np.exp(lflat)
        return np.exp(lamp) * generalized_normal(tw, loc, np.exp(lsc), shape) + np.exp(lflat)
    def nll(q):
        mu = np.maximum(model_(q), 1e-12)
        return float((mu - hw * np.log(mu)).sum())
    fw = max((h > peak / 2).sum() * dt / 2.355, dt)
    q0 = np.array([pk * dt, np.log(fw), 0.0, np.log(max(hw.sum() - flat0 * len(hw), 1.0)), np.log(flat0)])
    if pulse == 'emg':
        q0 = np.array([pk * dt - 0.2, np.log(0.1), np.log(0.4), q0[3], q0[4]])
    r = minimize(nll, q0, method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-6, fatol=1e-6))
    r = minimize(nll, r.x, method='BFGS')
    loc, lsc, shape, lamp, lflat = r.x
    mu = np.maximum(model_(r.x), 1e-12)
    dev = float(2.0 * (mu - hw + hw * np.log(np.maximum(hw, 1e-12) / mu)).sum())
    dof = max(len(hw) - 5, 1)
    resp = emg_pulse(t, loc, np.exp(lsc), np.exp(shape)) if pulse == 'emg' else generalized_normal(t, loc, np.exp(lsc), shape)
    return dict(response=resp, loc=float(loc), scale=float(np.exp(lsc)),
                shape=float(np.exp(shape) if pulse == 'emg' else shape), pulse=pulse,
                amplitude=float(np.exp(lamp)), flat=float(np.exp(lflat)), window=(a, b),
                dev=dev, dof=dof, dpd=dev / dof, peak_channel=pk)


def model(loaded=None, n_coef=25, which='h20', verbose=True,
          samples=('D0', 'A0', 'DA'), detectors=None, irf='h20', rebin=False, growth=1.05,
          rl_iterations=500, irf_conv_stop=None, peak_stop=1.0, peak_pulse='gn', rho_grid=None,
          d0_from=None, maps=None, mask_edges=None, sample_shifts=False, irf_by_sample=None):
    """Everything the fit needs: the maps on this axis, the measured responses,
    the twelve histograms, and the graph.

    The geometry is the one `experiment.py` calls interleaved: each histogram
    holds both pulse windows, so every (sample, detector) appears twice among
    the physics channels -- once for each pulse -- and the data live on the
    green-pulse key with the red-pulse partner summed into the same mean.
    """
    import torch
    irf_kind = irf                 # `irf` is rebound to the responses below
    peak_fits = {}
    d = loaded or load()
    cal = d['cal']
    E, rel, spl, L = environment(n_coef=n_coef, verbose=verbose, rho_grid=rho_grid, maps=maps)
    n = int(E['n'])
    irf, info, off = responses(d, which=which, n=n)
    irf_h20 = {k: v.copy() for k, v in irf.items()}
    y_np, mask_np = histograms(d, n=n, mask_edges=mask_edges)
    if d0_from is not None:
        #: THE REFERENCE DYE AS THE DONOR-ONLY SAMPLE (tpeulen, prompt 418: "try
        #: Rh110 and see if you can get l1,l2").  A free dye is one lifetime and
        #: a rotation of a few hundred picoseconds, so what its two polarised
        #: channels disagree about after that is the detection: g, l1 and l2.
        #: Its histograms stand in for the D0 sample's under the D0 scope --
        #: same physics, no distance -- and its decay wraps the period, which
        #: the periodic convolution carries.
        for det, (colour, pol) in DETECTORS.items():
            k = ('D0', f'{det}_{KIND[det]}')
            if k in y_np:
                y_np[k] = np.asarray(d['reference'][(d0_from, colour, pol)], float)[:n]

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
    elif irf_kind == 'peakfit':
        #: chisurf's skewed Gaussian fitted to the water measurement's peak
        #: region, per detector, on the pulse `responses()` selected for it;
        #: the fitted shape over the water response's own support is the
        #: response, its tail the shape's and not the dirt's
        peak_fits = {}
        for det in dets:
            w = np.asarray(irf_h20[det], float)
            pf = peak_fit_response(w, cal['dt'], stop_after_peak=peak_stop, pulse=peak_pulse)
            f = np.where(w > 0, pf['response'], 0.0)
            irf[det] = f * (w.sum() / max(f.sum(), 1e-300))
            info[det] = dict(info[det], source=f"skewed Gaussian on the water peak (to {peak_stop:g} ns after it)",
                             background_fraction=0.001, peak_fit=pf)
            peak_fits[det] = pf
    elif irf_kind not in ('h20', 'analytic'):
        raise ValueError(f"irf must be 'h20', 'rh110', 'analytic' or 'peakfit', not {irf_kind!r}")
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
    #: R1(r): a water measurement per sample where the default does not belong
    measured_by_sample, irf_by_sample_resp = {}, {}
    for samp_, which_ in (irf_by_sample or {}).items():
        if samp_ not in samples or which_ == which:
            continue
        irf_s, info_s, _ = responses(d, which=which_, n=n)
        for det in dets:
            measured_by_sample[(samp_, det)] = L.tt(irf_s[det])
            irf_by_sample_resp[(samp_, det)] = irf_s[det]
            irf_bg_med[f'{det}_{samp_}'] = max(min(info_s[det]['background_fraction'], 0.9), 1e-3)

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
                            irf_shape=analytic, bkg_medians=bkg_med, ref_dets=ref_dets, sample_shifts=sample_shifts,
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
        L.InstrumentModel(E, 'measured', {det: L.tt(irf[det]) for det in dets},
                          measured_by_sample=measured_by_sample)
    ps = L.PSplineFactor(n_coef, spl=spl)
    g = L.FactorGraph(E, keys, V, L.PoissonCountsFactor({k: y[k] for k in pairs}, mask=masks),
                      inst, spl, ps)
    g.rel = rel
    g.start_y = {k: y[k] * E['win_green'] for k in pairs}
    return dict(L=L, E=E, Ep=E, rel=rel, spl=spl, keys=keys, pairs=pairs, graph=g,
                y=y, y_full=y_full, t=t_axis, masks=masks, irf=irf, irf_info=info, offset_channels=off,
                irf_kind=irf_kind, peak_fits=peak_fits, irf_by_sample=irf_by_sample_resp,
                irf_h20=irf_h20,
                cal=cal, loaded=d, n=n, n_coef=n_coef,
                bkg_medians=bkg_med, scale_medians=scale_med, irf_bg_medians=irf_bg_med)


def fit(m, lam_nodes=(1.0, 0.0, -1.0), seed=0, verbose=True, accelerate=False, fixed=None,
        spec_smoothness=None):
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
    L.Laplace.analytic = True          # the REF blocks are in it (R1(g)-2: 3.9e-16 against AD, 18 ms against 3.6 s)
    #: START FROM THE MEASURED BACKGROUND.  `start_from_data` fits it by a
    #: non-negative solve on a 33-column basis whose long-lifetime members are
    #: nearly identical, and on this measurement it returned 0.0714 and 0.0000
    #: for two channels that must agree -- 48 counts per channel into a
    #: histogram whose flat parts say 15.  Notebook 19 measures it directly, so
    #: the start uses that and the fit refines it under a prior a sixth of a
    #: decade wide.  Worth 18.6 to 13.0 in the deviance at the start, and the
    #: difference between a fit that converges and one that does not.
    #: THE LIFETIME SPECTRA'S CONTINUITY (R3, 2026-09-14): free log amplitudes
    #: on the lifetime grid are flat, non-concave directions and no node
    #: converged without it.  'evidence' chooses lambda_s on the donor-only
    #: decay alone, as the prototype's `fit_sample` does, and holds it.
    m['spec_smooth_evidence'] = None
    if spec_smoothness == 'evidence':
        ss, ev_s = L.choose_spectrum_smoothness(m['graph'], m['y'])
        m['graph'].spec_smooth = ss; m['spec_smooth_evidence'] = ev_s
        if verbose:
            print('  spectrum smoothness by evidence on D0: ' + ', '.join(f'{k:g}: {v:.1f}' for k, v in ev_s.items())
                  + f' -> log10 lambda_s {math.log10(ss.lam_s) if ss else float("nan"):g}')
    elif spec_smoothness is not None:
        m['graph'].spec_smooth = L.SpectrumSmoothness(m['E']['Kint'], lam_s=10.0 ** float(spec_smoothness), weak_sd=3.0)
    th0, _ = L.start_from_data(m['graph'], m['y'], verbose=False, method='mem')
    for k, med in m['bkg_medians'].items():
        nm = f'bkg_{k[0]}_{k[1]}'
        if nm in m['graph'].offsets:
            a, b = m['graph'].offsets[nm]
            th0[a:b] = m['graph'].index[nm].transform.to_unconstrained(L.tt([med]))
    if 'spec_ref_eps' in m['graph'].offsets:          # the 'free' reference spectrum only
        #: the reference dye's spectrum starts where the donor's does -- a
        #: start, not a prior; `start_from_data` knows only the donor's
        a, b = m['graph'].offsets['spec_ref_eps']; a0, b0 = m['graph'].offsets['spec_eps']
        th0[a:b] = th0[a0:b0]
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
        #: `restrict(parent, theta_of_parent)`: the PARENT graph's theta mapped
        #: onto the node graph by name. It was `gi.restrict(gi, th_prev)` --
        #: the node graph as its own parent -- which reads the parent's theta
        #: with the node graph's offsets and scrambles every variable: a start
        #: at D/dof 110 became 451,973 on the node graph (2026-09-14, found on
        #: Rh110). Every fit through here started from that, which is the
        #: "walk" R3 was written for: runaway shifts, competing modes, fits
        #: that converged to nonsense.
        r = L.laplace_at(gi, gi.restrict(g, th_prev), optimiser='fisher',
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


def poisson_reference_by_histogram(post, n_draw=150, seed=0):
    """The measured deviance reference, overall AND per histogram, from the
    same draws (A0 of the plan, 2026-09-14): Poisson data at the fitted means,
    their deviance against those means.  `pie_mfd.poisson_reference` sums the
    histograms inside each draw; this keeps them apart, so a z per histogram
    says which one carries the misfit.  The overall numbers equal
    `poisson_reference(post, n_draw, seed)` (same generator, same order)."""
    rng = np.random.default_rng(seed)
    lam = {k: np.asarray(v) for k, v in post['lam'].items()}
    sel = {k: post['rows'][k]['sel'] for k in lam}
    per = {k: [] for k in lam}; tot = []
    for _ in range(n_draw):
        t_all = 0.0
        for k, l in lam.items():
            mu = l[sel[k]]
            yk = rng.poisson(mu)
            with np.errstate(divide='ignore', invalid='ignore'):
                t = np.where(yk > 0, yk * np.log(np.maximum(yk, 1e-300) / mu), 0.0)
            dk = 2.0 * float((t - (yk - mu)).sum())
            per[k].append(dk / post['rows'][k]['dof']); t_all += dk
        tot.append(t_all / post['dof'])
    tot = np.array(tot)
    return (float(tot.mean()), float(tot.std())), \
        {k: (float(np.mean(v)), float(np.std(v))) for k, v in per.items()}


def rule0(m, post):
    """Rule 0 per histogram: the Poisson deviance per degree of freedom and a
    runs test on the weighted residuals, with the deviance compared against a
    reference MEASURED by drawing Poisson data at the fitted means -- overall
    and for each histogram, with the share of the total excess it carries."""
    ref, ref_k = poisson_reference_by_histogram(post, n_draw=150, seed=0)
    excess = {k: r['dev'] - ref_k[k][0] * r['dof'] for k, r in post['rows'].items()}
    ex_tot = sum(max(e, 0.0) for e in excess.values()) or 1.0
    rows = []
    for k, r in post['rows'].items():
        mu, sd = ref_k[k]
        rows.append(dict(channel=f'{k[0]} {k[1]}', key=k, counts=float(m['y'][k].sum()),
                         dpd=r['dpd'], runs_p=r['runs_p'], ref=mu, ref_sd=sd,
                         z=float((r['dpd'] - mu) / max(sd, 1e-12)),
                         share=float(max(excess[k], 0.0) / ex_tot)))
    dpd = post['dev'] / post['dof']
    rows.append(dict(channel='all', key=None, counts=float(sum(float(m['y'][k].sum()) for k in m['pairs'])),
                     dpd=dpd, runs_p=float('nan'), ref=ref[0], ref_sd=ref[1],
                     z=float((dpd - ref[0]) / max(ref[1], 1e-12)), share=1.0))
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


def apply_physical_bounds(m, r0_max=None, l_bounds=None, verbose=True):
    """Bound what physics bounds, instead of letting the fit use it as a knob
    (A3 of the plan, 2026-09-14).  The configuration of record put `l1` at
    -0.117 (a mixing fraction cannot be negative), `r0_ref` at 0.445 (a dye's
    fundamental anisotropy cannot exceed 0.4) and `r0_d` on its lower bound.

    `r0_max`: both fundamental anisotropies on Logit(0.15, r0_max) with a
    uniform prior.  `l_bounds=(lo, hi)`: l1 and l2 on Logit(lo, hi), uniform.
    Replaces the variables' transforms and priors in place, before any start
    or fit is computed from them."""
    L = m['L']; changed = []
    for v in m['graph'].V:
        if r0_max is not None and v.name in ('r0_d', 'r0_ref'):
            v.transform = L.Logit(0.15, float(r0_max)); v.prior = L.Uniform(0.15, float(r0_max))
            changed.append((v.name, 0.15, float(r0_max)))
        if l_bounds is not None and v.name in ('l1', 'l2'):
            lo, hi = map(float, l_bounds)
            v.transform = L.Logit(lo, hi); v.prior = L.Uniform(lo, hi)
            changed.append((v.name, lo, hi))
    if verbose and changed:
        print('physical bounds: ' + ', '.join(f'{n} in [{a:g}, {b:g}]' for n, a, b in changed))
    return changed


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
