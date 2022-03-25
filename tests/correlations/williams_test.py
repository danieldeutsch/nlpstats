import numpy as np
import pytest
import unittest

from nlpstats.correlations import williams_test


class TestWilliams(unittest.TestCase):
    def test_williams_test(self):
        # This test verifies that the output is the same as the psych package for
        # several different randomly generated inputs
        n, m = 9, 5

        np.random.seed(12)
        X = np.random.random((n, m))
        Y = np.random.random((n, m))
        Z = np.random.random((n, m))

        # These are used as input to r.test
        # effective_N = n, m
        # r12 = corr_func(X, Z)
        # r13 = corr_func(Y, Z)
        # r23 = corr_func(X, Y)

        # One tail
        result = williams_test(X, Y, Z, "global", "pearson", alternative="greater")
        self.assertAlmostEqual(result.pvalue, 0.2716978, places=5)

        # The opposite order should produce 1-0.2716978. r.test does not do this and
        # will return 0.2716978 because it assumes that r12 > r13.
        result = williams_test(Y, X, Z, "global", "pearson", alternative="greater")
        self.assertAlmostEqual(result.pvalue, 1.0 - 0.2716978, places=5)

        # Changing the alternative should also reverse them
        result = williams_test(X, Y, Z, "global", "pearson", alternative="less")
        self.assertAlmostEqual(result.pvalue, 1.0 - 0.2716978, places=5)

        result = williams_test(Y, X, Z, "global", "pearson", alternative="less")
        self.assertAlmostEqual(result.pvalue, 0.2716978, places=5)

        # Two tails
        result = williams_test(X, Y, Z, "global", "pearson")
        self.assertAlmostEqual(result.pvalue, 0.5433956, places=5)

        # Should not matter the order for two tails
        result = williams_test(Y, X, Z, "global", "pearson")
        self.assertAlmostEqual(result.pvalue, 0.5433956, places=5)

        X = np.random.random((n, m))
        Y = np.random.random((n, m))
        Z = np.random.random((n, m))

        # These are used as input to r.test
        # effective_N = N
        # r12 = corr_func(X, Z)
        # r13 = corr_func(Y, Z)
        # r23 = corr_func(X, Y)

        # One tail
        # Since r12 < r13, r.test will only replicate this result with the reversed input order
        result = williams_test(Y, X, Z, "system", "spearman", alternative="greater")
        self.assertAlmostEqual(result.pvalue, 0.4658712, places=5)

        # r.test would return the same result here, but we return 1.0 - expected
        result = williams_test(X, Y, Z, "system", "spearman", alternative="greater")
        self.assertAlmostEqual(result.pvalue, 1.0 - 0.4658712, places=5)

        # Changing the alternative should swap the pvalues
        result = williams_test(Y, X, Z, "system", "spearman", alternative="less")
        self.assertAlmostEqual(result.pvalue, 1.0 - 0.4658712, places=5)

        # r.test would return the same result here, but we return 1.0 - expected
        result = williams_test(X, Y, Z, "system", "spearman", alternative="less")
        self.assertAlmostEqual(result.pvalue, 0.4658712, places=5)

        # Two tails
        result = williams_test(X, Y, Z, "system", "spearman")
        self.assertAlmostEqual(result.pvalue, 0.9317423, places=5)

        # Order doesn't matter
        result = williams_test(Y, X, Z, "system", "spearman")
        self.assertAlmostEqual(result.pvalue, 0.9317423, places=5)

    def test_williams_test_iv(self):
        X = np.random.random((9, 5))
        Y = np.random.random((9, 5))
        Z = np.random.random((9, 5))

        message = "Unknown correlation level"
        with pytest.raises(ValueError, match=message):
            williams_test(Y, X, Z, "UNK", "spearman")

        message = "Unknown alternative"
        with pytest.raises(ValueError, match=message):
            williams_test(Y, X, Z, "system", "spearman", alternative="UNK")
