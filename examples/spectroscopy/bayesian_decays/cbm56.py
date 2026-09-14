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
           'period_from_data', 'background', 'count_rates', 'pulse_positions',
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
