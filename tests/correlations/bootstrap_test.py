import numpy as np
import pytest
import unittest

from nlpstats.correlations.bootstrap import bootstrap


class TestBootstrap(unittest.TestCase):
    @pytest.mark.filterwarnings("ignore: An input array is constant")
    def test_bootstrap_regression(self):
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
