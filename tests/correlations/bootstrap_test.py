import numpy as np
import pytest
import unittest

from nlpstats.correlations.bootstrap import bootstrap


class TestBootstrap(unittest.TestCase):
    @pytest.mark.filterwarnings("ignore: An input array is constant")
    def test_bootstrap_regression_small(self):
        np.random.seed(3)
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        Y = np.array([[5, 2, 7], [1, 7, 3], [4, 2, 2]])

        result = bootstrap(X, Y, "global", "pearson", "systems")
        self.assertAlmostEqual(result.lower, -0.8660254037844388, places=4)
        self.assertAlmostEqual(result.upper, 0.39735970711951324, places=4)

        result = bootstrap(X, Y, "global", "pearson", "systems", confidence_level=0.9)
        self.assertAlmostEqual(result.lower, -0.5773502691896258, places=4)
        self.assertAlmostEqual(result.upper, 0.32732683535398865, places=4)

        result = bootstrap(X, Y, "global", "pearson", "inputs")
        self.assertAlmostEqual(result.lower, -0.9449111825230679, places=4)
        self.assertAlmostEqual(result.upper, 0.0, places=4)

        result = bootstrap(X, Y, "global", "pearson", "both")
        self.assertAlmostEqual(result.lower, -1.0, places=4)
        self.assertAlmostEqual(result.upper, 1.0, places=4)

    @pytest.mark.filterwarnings("ignore: An input array is constant")
    def test_bootstrap_regression_random(self):
        np.random.seed(4)
        X = np.random.rand(25, 50)
        Y = np.random.rand(25, 50)

        result = bootstrap(X, Y, "system", "pearson", "both")
        self.assertAlmostEqual(result.lower, -0.5297912355172001, places=4)
        self.assertAlmostEqual(result.upper, 0.5460669817957821, places=4)

        result = bootstrap(X, Y, "input", "spearman", "inputs", n_resamples=100)
        self.assertAlmostEqual(result.lower, -0.04338807692307692, places=4)
        self.assertAlmostEqual(result.upper, 0.05723730769230765, places=4)

        result = bootstrap(X, Y, "global", "kendall", "systems")
        self.assertAlmostEqual(result.lower, -0.0347037143864372, places=4)
        self.assertAlmostEqual(result.upper, 0.028916791839461616, places=4)

    def test_bootstrap_iv(self):
        X = np.random.rand(3, 4)
        Y = np.random.rand(3, 4)

        message = "`paired_inputs` must be `True` for input-level correlations"
        with pytest.raises(ValueError, match=message):
            bootstrap(X, Y, "input", "pearson", "both", paired_inputs=False)

        message = "`confidence_level` must be between 0 and 1"
        with pytest.raises(ValueError, match=message):
            bootstrap(X, Y, "input", "pearson", "both", confidence_level=0.0)
        with pytest.raises(ValueError, match=message):
            bootstrap(X, Y, "input", "pearson", "both", confidence_level=1.0)
        with pytest.raises(ValueError, match=message):
            bootstrap(X, Y, "input", "pearson", "both", confidence_level=-0.1)
        with pytest.raises(ValueError, match=message):
            bootstrap(X, Y, "input", "pearson", "both", confidence_level=1.1)

        message = "`n_resamples` must be a positive integer"
        with pytest.raises(ValueError, match=message):
            bootstrap(X, Y, "input", "pearson", "both", n_resamples=0)
        with pytest.raises(ValueError, match=message):
            bootstrap(X, Y, "input", "pearson", "both", n_resamples=-1)