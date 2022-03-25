import numpy as np
import pytest
import unittest

from nlpstats.correlations import permutation_test


class TestPermutation(unittest.TestCase):
    def test_permutation_test_regression_random(self):
        np.random.seed(1234)
        X = np.random.random((9, 5))
        Y = np.random.random((9, 5))
        Z = np.random.random((9, 5))

        result = permutation_test(X, Y, Z, "global", "pearson", "both")
        self.assertAlmostEqual(result.pvalue, 0.5361536153615362, places=4)

        result = permutation_test(
            X,
            Y,
            Z,
            "global",
            "pearson",
            "both",
            alternative="greater",
            n_resamples=1000,
        )
        self.assertAlmostEqual(result.pvalue, 0.741, places=4)

        result = permutation_test(
            X, Y, Z, "global", "pearson", "both", alternative="less", n_resamples=1000
        )
        self.assertAlmostEqual(result.pvalue, 0.299, places=4)

        result = permutation_test(
            X, Y, Z, "system", "spearman", "inputs", n_resamples=1000
        )
        self.assertAlmostEqual(result.pvalue, 0.698, places=4)

        result = permutation_test(
            X, Y, Z, "input", "kendall", "systems", n_resamples=1000
        )
        self.assertAlmostEqual(result.pvalue, 0.519, places=4)

        # Unpaired inputs
        X = np.random.random((9, 30))
        Y = np.random.random((9, 30))

        result = permutation_test(
            X, Y, Z, "system", "kendall", "both", n_resamples=1000, paired_inputs=False
        )
        self.assertAlmostEqual(result.pvalue, 0.126, places=4)

    def test_permutation_test_iv(self):
        X = np.random.random((9, 5))
        Y = np.random.random((9, 5))
        Z = np.random.random((9, 5))

        message = (
            "`paired_inputs` must be `True` for input- or global-level correlations"
        )
        with pytest.raises(ValueError, match=message):
            permutation_test(X, Y, Z, "input", "pearson", "both", paired_inputs=False)
        with pytest.raises(ValueError, match=message):
            permutation_test(X, Y, Z, "global", "pearson", "both", paired_inputs=False)

        message = "Unknown alternative"
        with pytest.raises(ValueError, match=message):
            permutation_test(X, Y, Z, "global", "pearson", "both", alternative="UNK")

        message = "`n_resamples` must be a positive integer"
        with pytest.raises(ValueError, match=message):
            permutation_test(X, Y, Z, "global", "pearson", "both", n_resamples=0)
        with pytest.raises(ValueError, match=message):
            permutation_test(X, Y, Z, "global", "pearson", "both", n_resamples=-1)
