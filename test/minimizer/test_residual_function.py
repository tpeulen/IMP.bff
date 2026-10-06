"""FitMinimizer over a residual *function*: the scipy.optimize.least_squares shape.

A FitResidualFunction subclass replaces ports + graph objective, so a caller
with an arbitrary model (a Python closure) gets bff's bounded MINPACK
Levenberg-Marquardt directly. scipy is the oracle where it is installed.
"""

import gc
import unittest

import numpy as np

from IMP import bff

try:
    import scipy.optimize as so
except ImportError:  # pragma: no cover
    so = None


class Exponential(bff.FitResidualFunction):
    def __init__(self, t, y):
        super().__init__()
        self.t, self.y = t, y

    def evaluate(self, p):
        return p[0] * np.exp(-self.t / p[1]) + p[2] - self.y


class Raises(bff.FitResidualFunction):
    def evaluate(self, p):
        raise ZeroDivisionError("from the residual function")


def _data(seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10, 150)
    return t, 3.0 * np.exp(-t / 1.7) + 0.2 + rng.normal(scale=0.01, size=t.size)


def _minimizer(fn, x0, lb, ub):
    m = bff.FitMinimizer()
    m.set_residual_function(fn)
    m.set_initial_values(x0)
    m.set_bounds(lb, ub)
    return m


class TestResidualFunction(unittest.TestCase):
    def test_callback_parameters_are_owned_numeric_arrays(self):
        """A callback may retain parameters after later evaluations and C++ cleanup."""
        class Retaining(bff.FitResidualFunction):
            def __init__(self):
                super().__init__()
                self.parameters = []
                self.expected = []

            def evaluate(self, parameters):
                """Retain the numeric input while returning a simple residual."""
                if not isinstance(parameters, np.ndarray):
                    raise TypeError("callback parameters must be a numeric ndarray")
                self.parameters.append(parameters)
                self.expected.append(parameters.copy())
                return np.array([parameters[0] - 2.0, parameters[0] - 2.0])

        function = Retaining()
        minimizer = _minimizer(function, [1.0], [-np.inf], [np.inf])
        minimizer.run()
        retained, expected = function.parameters, function.expected
        self.assertGreater(len(retained), 1)
        del minimizer, function
        gc.collect()
        for parameters, snapshot in zip(retained, expected):
            np.testing.assert_array_equal(parameters, snapshot)
            self.assertEqual(parameters.dtype, np.float64)

    def test_none_clears_an_optional_residual_callback(self):
        """Clearing a callback remains distinct from supplying an empty parameter array."""
        minimizer = bff.FitMinimizer()
        minimizer.set_residual_function(Raises())
        minimizer.set_residual_function(None)
        self.assertIsNone(minimizer.get_residual_function())
        self.assertFalse(minimizer.has_objective())

    def test_a_callback_signature_error_keeps_its_python_exception(self):
        """An override requiring an extra argument reports TypeError across C++."""
        class WrongSignature(bff.FitResidualFunction):
            def evaluate(self, parameters, extra):
                return [parameters[0] - extra]

        minimizer = _minimizer(WrongSignature(), [1.0], [-np.inf], [np.inf])
        with self.assertRaises(TypeError):
            minimizer.run()

    def test_fits_and_converges(self):
        t, y = _data()
        m = _minimizer(Exponential(t, y), [1.0, 1.0, 0.0], [0, 0.1, -1], [10, 10, 1])
        info = m.run()
        self.assertIn(info, (1, 2, 3, 4))
        np.testing.assert_allclose(m.get_x(), [3.0, 1.7, 0.2], atol=0.05)
        self.assertEqual(len(m.get_residuals()), t.size)
        self.assertEqual(len(m.get_covariance()), 9)

    @unittest.skipIf(so is None, "scipy is the oracle")
    def test_same_minimum_as_scipy(self):
        for seed in range(10):
            t, y = _data(seed)
            fn = Exponential(t, y)
            m = _minimizer(fn, [1.0, 1.0, 0.0], [0, 0.1, -1], [10, 10, 1])
            m.run()
            s = so.least_squares(fn.evaluate, [1.0, 1.0, 0.0], bounds=([0, 0.1, -1], [10, 10, 1]))
            np.testing.assert_allclose(m.get_x(), s.x, rtol=1e-4)
            self.assertLessEqual(0.5 * m.get_chi2(), s.cost * (1 + 1e-6))

    def test_jacobian_at_solution(self):
        t, y = _data()
        m = _minimizer(Exponential(t, y), [1.0, 1.0, 0.0], [0, 0.1, -1], [10, 10, 1])
        m.run()
        jac = np.asarray(m.compute_jacobian(m.get_x())).reshape(3, t.size)
        np.testing.assert_allclose(jac[2], 1.0, rtol=1e-6)

    def test_start_at_the_midpoint_of_a_box_moves(self):
        """A start at a two-sided box's midpoint is fitted, not returned.

        There the internal parameter is ~1e-17 (asin's rounding), so the
        forward-difference step ``eps * |xi|`` is ~1e-25: nonzero, yet it
        moves the parameter by nothing. The Jacobian column was zero and the
        gtol test ended the fit after two evaluations, at the start. A
        log-midpoint start (tau0 = sqrt(lo * hi)) hits this exactly.
        """
        t = np.linspace(0.0, 5.0, 512)
        y = 2.0 * np.exp(-t / 3.0)

        class LogTau(bff.FitResidualFunction):
            def evaluate(self, p):
                return 2.0 * np.exp(-t / np.exp(p[0])) - y

        lo, hi = np.log(0.5), np.log(6.0)
        m = _minimizer(LogTau(), [0.5 * (lo + hi)], [lo], [hi])
        m.run()
        self.assertAlmostEqual(m.get_x()[0], np.log(3.0), places=5)
        self.assertGreater(m.get_number_of_evaluations(), 2)

    def test_python_exception_propagates(self):
        m = _minimizer(Raises(), [1.0], [-np.inf], [np.inf])
        with self.assertRaises(ZeroDivisionError):
            m.run()

    def test_function_and_objective_are_exclusive(self):
        t, y = _data()
        m = bff.FitMinimizer()
        m.set_residual_function(Exponential(t, y))
        self.assertTrue(m.has_objective())
        self.assertIsNotNone(m.get_residual_function())


if __name__ == "__main__":
    unittest.main()
