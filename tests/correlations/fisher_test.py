import numpy as np
import pytest
import unittest

from nlpstats.correlations import fisher


class TestFisher(unittest.TestCase):
    def test_fisher_regression(self):
        np.random.seed(12)
        X = np.random.rand(5, 7)
        Y = np.random.rand(5, 7)

        result = fisher(X, Y, "global", "pearson")
        self.assertAlmostEqual(result.lower, -0.02763744135012373)
        self.assertAlmostEqual(result.upper, 0.5818846438651135)

        result = fisher(X, Y, "global", "spearman")
        self.assertAlmostEqual(result.lower, -0.06733469087453943)
        self.assertAlmostEqual(result.upper, 0.5640758668009686)

        result = fisher(X, Y, "global", "kendall")
        self.assertAlmostEqual(result.lower, -0.029964677270600665)
        self.assertAlmostEqual(result.upper, 0.4098565164085108)

        result = fisher(X, Y, "system", "pearson")
        self.assertAlmostEqual(result.lower, -0.6445648014599665)
        self.assertAlmostEqual(result.upper, 0.9644395142168088)

        result = fisher(X, Y, "system", "spearman")
        self.assertAlmostEqual(result.lower, -0.6708734441360908)
        self.assertAlmostEqual(result.upper, 0.9756771001362685)

        result = fisher(X, Y, "system", "kendall")
        self.assertAlmostEqual(result.lower, -0.7023910748254728)
        self.assertAlmostEqual(result.upper, 0.9377789575997956)

        result = fisher(X, Y, "input", "pearson")
        self.assertAlmostEqual(result.lower, -0.808376631595968)
        self.assertAlmostEqual(result.upper, 0.9287863878043723)

        result = fisher(X, Y, "input", "spearman")
        self.assertAlmostEqual(result.lower, -0.7262127280589684)
        self.assertAlmostEqual(result.upper, 0.9653646507719408)

        result = fisher(X, Y, "input", "kendall")
        self.assertAlmostEqual(result.lower, -0.684486849088761)
        self.assertAlmostEqual(result.upper, 0.9418063314024349)

    def test_fisher_iv(self):
        X = np.random.rand(5, 7)
        Y = np.random.rand(5, 7)

        message = "`confidence_level` must be between 0 and 1"
        with pytest.raises(ValueError, match=message):
            fisher(X, Y, "input", "pearson", confidence_level=0.0)
        with pytest.raises(ValueError, match=message):
            fisher(X, Y, "input", "pearson", confidence_level=1.0)
        with pytest.raises(ValueError, match=message):
            fisher(X, Y, "input", "pearson", confidence_level=-0.1)
        with pytest.raises(ValueError, match=message):
            fisher(X, Y, "input", "pearson", confidence_level=1.1)

        message = "Unknown correlation coefficient"
        with pytest.raises(ValueError, match=message):
            fisher(X, Y, "input", "UNK")

        message = "Unknown correlation level"
        with pytest.raises(ValueError, match=message):
            fisher(X, Y, "UNK", "pearson")
