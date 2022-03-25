import numpy as np
import pytest
import unittest

from nlpstats.correlations.resampling import permute, resample


class TestResample(unittest.TestCase):
    def setUp(self) -> None:
        # fmt: off
        self.m = 4
        self.n = 3
        self.X = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])
        self.Y = np.array([
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
            [22, 23, 24]
        ])
        # fmt: on

    def test_resample_systems(self):
        # Set the random seed and ensure we know the expected rows
        # that should be sampled
        np.random.seed(2)
        rows = np.random.choice(self.m, self.m, replace=True)
        np.testing.assert_equal(rows, [0, 3, 1, 0])

        # Reset seed and resample
        np.random.seed(2)
        X_s = resample(self.X, "systems")
        np.testing.assert_equal(X_s, [[1, 2, 3], [10, 11, 12], [4, 5, 6], [1, 2, 3]])

        # Reset seed and resample multiple matrices
        np.random.seed(2)
        X_s, Y_s = resample((self.X, self.Y), "systems")
        np.testing.assert_equal(X_s, [[1, 2, 3], [10, 11, 12], [4, 5, 6], [1, 2, 3]])
        np.testing.assert_equal(
            Y_s, [[13, 14, 15], [22, 23, 24], [16, 17, 18], [13, 14, 15]]
        )

    def test_resample_inputs(self):
        # Set the random seed and ensure we know the expected columns
        # that should be sampled
        np.random.seed(2)
        cols = np.random.choice(self.n, self.n, replace=True)
        np.testing.assert_equal(cols, [0, 1, 0])

        # Reset seed and resample
        np.random.seed(2)
        X_s = resample(self.X, "inputs")
        np.testing.assert_equal(X_s, [[1, 2, 1], [4, 5, 4], [7, 8, 7], [10, 11, 10]])

        # Reset seed and resample multiple matrices
        np.random.seed(2)
        X_s, Y_s = resample((self.X, self.Y), "inputs")
        np.testing.assert_equal(X_s, [[1, 2, 1], [4, 5, 4], [7, 8, 7], [10, 11, 10]])
        np.testing.assert_equal(
            Y_s, [[13, 14, 13], [16, 17, 16], [19, 20, 19], [22, 23, 22]]
        )

        # Now with unpaired inputs
        np.random.seed(2)
        cols1 = np.random.choice(self.n, self.n, replace=True)
        cols2 = np.random.choice(self.n, self.n, replace=True)
        np.testing.assert_equal(cols1, [0, 1, 0])
        np.testing.assert_equal(cols2, [2, 2, 0])

        np.random.seed(2)
        X_s = resample(self.X, "inputs", paired_inputs=False)
        np.testing.assert_equal(X_s, [[1, 2, 1], [4, 5, 4], [7, 8, 7], [10, 11, 10]])

        np.random.seed(2)
        X_s, Y_s = resample((self.X, self.Y), "inputs", paired_inputs=False)
        np.testing.assert_equal(X_s, [[1, 2, 1], [4, 5, 4], [7, 8, 7], [10, 11, 10]])
        np.testing.assert_equal(
            Y_s, [[15, 15, 13], [18, 18, 16], [21, 21, 19], [24, 24, 22]]
        )

    def test_resample_both(self):
        # Set the random seed and ensure we know the expected rows and columns
        # that should be sampled
        np.random.seed(2)
        rows = np.random.choice(self.m, self.m, replace=True)
        cols = np.random.choice(self.n, self.n, replace=True)
        np.testing.assert_equal(rows, [0, 3, 1, 0])
        np.testing.assert_equal(cols, [2, 2, 0])

        # Reset seed and resample
        np.random.seed(2)
        X_s = resample(self.X, "both")
        np.testing.assert_equal(X_s, [[3, 3, 1], [12, 12, 10], [6, 6, 4], [3, 3, 1]])

        # Reset seed and resample multiple matrices
        np.random.seed(2)
        X_s, Y_s = resample((self.X, self.Y), "both")
        np.testing.assert_equal(X_s, [[3, 3, 1], [12, 12, 10], [6, 6, 4], [3, 3, 1]])
        np.testing.assert_equal(
            Y_s, [[15, 15, 13], [24, 24, 22], [18, 18, 16], [15, 15, 13]]
        )

        # Now with unpaired inputs
        np.random.seed(2)
        rows = np.random.choice(self.m, self.m, replace=True)
        cols1 = np.random.choice(self.n, self.n, replace=True)
        cols2 = np.random.choice(self.n, self.n, replace=True)
        np.testing.assert_equal(rows, [0, 3, 1, 0])
        np.testing.assert_equal(cols1, [2, 2, 0])
        np.testing.assert_equal(cols2, [2, 1, 1])

        np.random.seed(2)
        X_s = resample(self.X, "both", paired_inputs=False)
        np.testing.assert_equal(X_s, [[3, 3, 1], [12, 12, 10], [6, 6, 4], [3, 3, 1]])

        np.random.seed(2)
        X_s, Y_s = resample((self.X, self.Y), "both", paired_inputs=False)
        np.testing.assert_equal(X_s, [[3, 3, 1], [12, 12, 10], [6, 6, 4], [3, 3, 1]])
        np.testing.assert_equal(
            Y_s, [[15, 14, 14], [24, 23, 23], [18, 17, 17], [15, 14, 14]]
        )

    def test_resample_iv(self):
        X = [[1, 2, 3], [4, 5, 6]]
        message = "Input `matrices` must all be of type `np.ndarray`"
        with pytest.raises(TypeError, match=message):
            resample(X, "systems")

        X = np.array(X)
        message = "Unknown resampling method"
        with pytest.raises(ValueError, match=message):
            resample(X, "UNK")

    def test_permute_systems(self):
        # Set the random seed and ensure we know the expected rows
        # that will be permuted
        np.random.seed(5)
        rows = np.random.rand(self.m) > 0.5
        np.testing.assert_equal(rows, [0, 1, 0, 1])

        # Reset seed and permute
        np.random.seed(5)
        X_p, Y_p = permute(self.X, self.Y, "systems")
        np.testing.assert_equal(X_p, [[1, 2, 3], [16, 17, 18], [7, 8, 9], [22, 23, 24]])
        np.testing.assert_equal(
            Y_p, [[13, 14, 15], [4, 5, 6], [19, 20, 21], [10, 11, 12]]
        )

        # Ensure the original X and Y did not change
        np.testing.assert_equal(self.X, [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        np.testing.assert_equal(
            self.Y, [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]]
        )

    def test_permute_inputs(self):
        # Set the random seed and ensure we know the expected columns
        # that will be permuted
        np.random.seed(5)
        cols = np.random.rand(self.n) > 0.5
        np.testing.assert_equal(cols, [0, 1, 0])

        # Reset seed and permute
        np.random.seed(5)
        X_p, Y_p = permute(self.X, self.Y, "inputs")
        np.testing.assert_equal(X_p, [[1, 14, 3], [4, 17, 6], [7, 20, 9], [10, 23, 12]])
        np.testing.assert_equal(
            Y_p, [[13, 2, 15], [16, 5, 18], [19, 8, 21], [22, 11, 24]]
        )

        # Ensure the original X and Y did not change
        np.testing.assert_equal(self.X, [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        np.testing.assert_equal(
            self.Y, [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]]
        )

    def test_permute_both(self):
        # Set the random seed and ensure we know the expected rows
        # and columns that will be permuted
        np.random.seed(5)
        rows = np.random.rand(self.m) > 0.5
        cols = np.random.rand(self.n) > 0.5
        np.testing.assert_equal(rows, [0, 1, 0, 1])
        np.testing.assert_equal(cols, [0, 1, 1])

        # Reset seed and permute
        np.random.seed(5)
        X_p, Y_p = permute(self.X, self.Y, "both")
        np.testing.assert_equal(X_p, [[1, 14, 15], [16, 5, 6], [7, 20, 21], [22, 11, 12]])
        np.testing.assert_equal(
            Y_p, [[13, 2, 3], [4, 17, 18], [19, 8, 9], [10, 23, 24]]
        )

        # Ensure the original X and Y did not change
        np.testing.assert_equal(self.X, [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        np.testing.assert_equal(
            self.Y, [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]]
        )

    def test_permute_iv(self):
        X = np.random.rand(3, 4)
        Y = np.random.rand(3, 4)
        message = "Unknown permutation method"
        with pytest.raises(ValueError, match=message):
            permute(X, Y, "UNK")
