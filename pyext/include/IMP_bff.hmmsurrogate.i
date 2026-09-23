/*
 * The H2MM surrogate: a network that estimates HMM parameters from burst
 * features. Built only when IMP.bff links tttrlib.
 *
 * A tttrlib.HMM or tttrlib.HmmModel belongs to tttrlib's own SWIG module and
 * cannot be handed to this one, so the C++ members taking them are hidden and
 * Python goes through the array entry points instead. The helpers below
 * accept and return tttrlib objects by building them on the Python side;
 * tttrlib is imported inside them, never at module level, so importing
 * IMP.bff does not require it (test_tttrlib_is_optional.py).
 */
#if IMP_BFF_HAS_TTTRLIB

/* The options type is a default argument of HmmSurrogate.train, so SWIG has
   to know it first; the generated module list is alphabetical. */
%include "IMP_bff.neuralnettraining.i"

%{
#include <IMP/bff/HMMSurrogate.h>
%}

%template(VectorVectorInt64) std::vector<std::vector<long long> >;

IMP_SWIG_VALUE(IMP::bff, HmmSurrogateEstimate, HmmSurrogateEstimates);

%ignore IMP::bff::HmmSurrogate::extract_features;
%ignore IMP::bff::HmmSurrogate::predict;
%ignore IMP::bff::HmmSurrogate::encode;
%ignore IMP::bff::HmmSurrogate::decode;
%ignore IMP::bff::HmmSurrogate::generate_training_set;

%include "IMP/bff/HMMSurrogate.h"

%extend IMP::bff::HmmSurrogate {
  /* X then Y, both row-major, in one vector; generate_training_set splits. */
  static std::vector<double> _generate_training_set(int n_states, int n_streams,
                                                    int n_samples, int n_bursts,
                                                    int burst_len, double mean_dt,
                                                    int seed) {
    std::vector<double> X, Y;
    IMP::bff::HmmSurrogate::generate_training_set(n_states, n_streams, n_samples,
                                                  n_bursts, burst_len, mean_dt, seed,
                                                  X, Y);
    X.insert(X.end(), Y.begin(), Y.end());
    return X;
  }

  %pythoncode %{
    @staticmethod
    def _bursts(times, streams):
        import numpy as np
        t = [np.asarray(x, dtype=np.int64).tolist() for x in times]
        s = [np.asarray(x, dtype=np.int64).tolist() for x in streams]
        if len(t) != len(s):
            raise ValueError("times and streams must hold the same bursts")
        return t, s

    @staticmethod
    def _layout(hmm):
        """The CSR layout of a tttrlib.HMM, as plain lists."""
        import numpy as np
        if hmm.get_n_micro_bins() != 1:
            raise ValueError("HmmSurrogate: a micro-time (product-alphabet) HMM "
                             "is not what a surrogate is trained on")
        return (np.asarray(hmm.get_streams(), dtype=np.int64).tolist(),
                np.asarray(hmm.get_offsets(), dtype=np.int64).tolist(),
                np.asarray(hmm.get_gap_slot(), dtype=np.int64).tolist(),
                np.asarray(hmm.get_unique_dt(), dtype=np.int64).tolist(),
                int(hmm.get_n_streams()))

    @staticmethod
    def features_from_bursts(times, streams, n_streams):
        """Feature vector of the bursts `tttrlib.HMM.set_bursts` takes, as a
        float64 array of length N_FEATURES."""
        import numpy as np
        t, s = HmmSurrogate._bursts(times, streams)
        return np.asarray(
            HmmSurrogate.extract_features_from_bursts(t, s, int(n_streams)), dtype=float)

    @staticmethod
    def features(hmm):
        """Feature vector of a loaded `tttrlib.HMM`, as a float64 array."""
        import numpy as np
        return np.asarray(
            HmmSurrogate.extract_features_from_layout(*HmmSurrogate._layout(hmm)),
            dtype=float)

    @staticmethod
    def generate_training_set(n_states, n_streams, n_samples=250, n_bursts=150,
                              burst_len=80, mean_dt=4.0, seed=0):
        """Simulate labelled datasets; returns `(X, Y)`, 2D float64 arrays."""
        import numpy as np
        flat = np.asarray(HmmSurrogate._generate_training_set(
            n_states, n_streams, n_samples, n_bursts, burst_len, mean_dt, seed),
            dtype=float)
        nx = n_samples * HmmSurrogate.N_FEATURES
        n_y = HmmSurrogate.n_targets(n_states, n_streams)
        return (flat[:nx].reshape(n_samples, HmmSurrogate.N_FEATURES),
                flat[nx:].reshape(n_samples, n_y))

    @staticmethod
    def to_hmm_model(estimate):
        """A `tttrlib.HmmModel` from an `HmmSurrogateEstimate`."""
        import tttrlib
        m = tttrlib.HmmModel(tttrlib.VectorDouble(list(estimate.get_prior())),
                             tttrlib.VectorDouble(list(estimate.get_trans())),
                             tttrlib.VectorDouble(list(estimate.get_obs())))
        m.n_phot = int(estimate.get_n_photons())
        return m

    def predict_model(self, times, streams):
        """Estimate a `tttrlib.HmmModel` from bursts in one forward pass.

        `times` and `streams` are what `tttrlib.HMM.set_bursts` takes: per
        burst, the macro times and the stream index of each photon. The
        result can be handed to `tttrlib.HMM.optimize` or `evaluate`.
        """
        t, s = HmmSurrogate._bursts(times, streams)
        return HmmSurrogate.to_hmm_model(self.predict_from_bursts(t, s))

    def predict_hmm(self, hmm):
        """Estimate a `tttrlib.HmmModel` from a loaded `tttrlib.HMM`."""
        return HmmSurrogate.to_hmm_model(
            self.predict_from_layout(*HmmSurrogate._layout(hmm)))

    def __repr__(self):
        return "HmmSurrogate(n_states={}, n_streams={}, features_version={})".format(
            self.get_n_states(), self.get_n_streams(), self.get_features_version())
  %}
}

#endif
