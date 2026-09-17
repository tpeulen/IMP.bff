"""
Kinetic schemes from factors
============================
Build the joint rate scheme of a dye whose photophysics depends on the
conformation of the protein it labels -- the scheme a PET or PIFE FCS
measurement sees -- from one small matrix per process, instead of writing the
six-state matrix out by hand. The conformational exchange and the
singlet-triplet photophysics are separate variables; the S1 decay is
conditioned on the conformation (quenched when closed). Amalgamation gives
the joint generator ``K[target, source]`` that the saturated FCS model takes.
"""

import numpy as np

import IMP.bff as bff

# Two clocks: spontaneous ("dark") rates, and excitation-scaled cross sections.
dark, excitation = bff.KineticNetwork(), bff.KineticNetwork()
for network in (dark, excitation):
    network.add_variable("conformation", 2)  # open, closed
    network.add_variable("photophysics", 3)  # S0, S1, T
    network.add_arc("conformation", "photophysics")

dark.set_rate("conformation", 0, 1, 800.0)  # open -> closed, Hz
dark.set_rate("conformation", 1, 0, 300.0)
for closed, k_s1 in ((0, 2.5e8), (1, 1.0e9)):
    dark.set_rate("photophysics", 1, 0, k_s1, [closed])  # S1 -> S0
    dark.set_rate("photophysics", 1, 2, 2.5e6, [closed])  # S1 -> T
    dark.set_rate("photophysics", 2, 0, 5.0e5, [closed])  # T -> S0
    excitation.set_rate("photophysics", 0, 1, 1.0, [closed])  # S0 -> S1

n = dark.get_number_of_states()
k_dark = np.asarray(dark.get_generator()).reshape(n, n)
k_exc = np.asarray(excitation.get_generator()).reshape(n, n)
print("joint states (conformation, photophysics):", [tuple(dark.get_states(i)) for i in range(n)])
print("K_dark[target, source] (Hz):")
print(k_dark)

# Where the molecule is at an excitation rate of 5e5 per second.
k_exc_0 = 5e5
p = np.asarray(bff.kinetic_stationary_distribution((k_dark + k_exc_0 * k_exc).ravel()))
print("closed fraction:", dark.get_marginal(p, "conformation")[1])
print("triplet fraction:", dark.get_marginal(p, "photophysics")[2])

# The same matrices drive the photokinetic bunching of the FCS curve.
brightness = [0.0, 1.0, 0.0, 0.0, 0.3, 0.0]  # S1 emits, dimmer when closed
tau = np.logspace(-8, -1, 8)
x = bff.fcs_bunching_factor(tau, k_exc_0, k_dark.ravel(), k_exc.ravel(), n, brightness)
for t, v in zip(tau, x):
    print("tau %8.1e s   X %.4f" % (t, v))
