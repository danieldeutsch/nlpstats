import numpy as np
import pytest
import unittest

from nlpstats.correlations.correlations import (
    correlate,
    global_level,
    input_level,
    system_level,
)


class TestCorrelations(unittest.TestCase):
    def test_system_level(self):
        # fmt: off
        X = np.array([
            [1, 9, 2],
            [4, 5, 2],
            [6, 7, 2]
        ])
        Y = np.array([
            [11, 12, 13],
            [14, 15, 16],
            [17, 18, 19]
        ])
        # fmt: on
        r = system_level(X, Y, "pearson")
        self.assertAlmostEqual(r, 0.7205766921, places=4)

    def test_system_level_nans(self):
        # fmt: off
        X = np.array([
            [1, 9, 2],
            [4, 5, np.nan],
            [6, np.nan, 2]
        ])
        Y = np.array([
            [11, 12, 13],
            [14, 15, np.nan],
            [17, np.nan, 19]
        ])
        # fmt: on
        r = system_level(X, Y, "pearson")
        self.assertAlmostEqual(r, -0.09578262852, places=4)

    @pytest.mark.filterwarnings("ignore: An input array is constant")
    def test_system_level_nan_correlation(self):
        # This shouldn't have any correlations because the average of
        # X is all the same
        # fmt: off
        X = np.array([
            [1, 2],
            [1, 2],
        ])
        Y = np.array([
            [11, 12],
            [14, 15],
        ])
        # fmt: on
        r = system_level(X, Y, "pearson")
        assert np.isnan(r)

    def test_system_level_custom_coefficient(self):
        # fmt: off
        X = np.array([
            [1, 3, 2],
            [4, 6, 2],
            [6, 7, 5]
        ])
        Y = np.array([
            [11, 12, 13],
            [14, 15, 16],
            [17, 18, 19]
        ])
        # fmt: on

        def _coef(x: np.ndarray, z: np.ndarray):
            return np.mean(x) - np.mean(z)

        r = system_level(X, Y, _coef)
        assert r == -11

    def test_system_level_iv(self):
        # One dimensional inputs
        # fmt: off
        X = np.array(
            [1, 9, 2]
        )
        Y = np.array([
            [11, 12, 13],
            [14, 15, np.nan],
            [17, np.nan, 19]
        ])
        # fmt: on
        message = "`X` must be two-dimensional"
        with pytest.raises(ValueError, match=message):
            system_level(X, Y, "pearson")

        message = "`Z` must be two-dimensional"
        with pytest.raises(ValueError, match=message):
            system_level(Y, X, "pearson")

        # Different number of rows
        # fmt: off
        X = np.array([
            [1, 9, 2],
            [8, 2, 3],
        ])
        Y = np.array([
            [11, 12, 13],
            [14, 15, np.nan],
            [17, np.nan, 19]
        ])
        # fmt: on
        message = "`X` and `Z` must have the same number of rows"
        with pytest.raises(ValueError, match=message):
            system_level(Y, X, "pearson")

        # Rows of all NaN
        # fmt: off
        X = np.array([
            [1, 9, 2],
            [np.nan, np.nan, np.nan],
            [8, 2, 1]
        ])
        Y = np.array([
            [11, 12, 13],
            [14, 15, np.nan],
            [17, np.nan, 19]
        ])
        # fmt: on
        message = "`X` must not have a row of all NaN"
        with pytest.raises(ValueError, match=message):
            system_level(X, Y, "pearson")

        message = "`Z` must not have a row of all NaN"
        with pytest.raises(ValueError, match=message):
            system_level(Y, X, "pearson")

        # Invalid correlation coefficient
        # fmt: off
        X = np.array([
            [1, 9, 2],
            [4, 5, 2],
            [6, 7, 2]
        ])
        Y = np.array([
            [11, 12, 13],
            [14, 15, 16],
            [17, 18, 19]
        ])
        # fmt: on
        message = "Unknown correlation coefficient"
        with pytest.raises(ValueError, match=message):
            system_level(Y, X, "DNE")

    def test_input_level(self):
        # fmt: off
        X = np.array([
            [1, 9, 2],
            [4, 5, 4],
            [6, 7, 1]
        ])
        Y = np.array([
            [11, 12, 13],
            [14, 15, 16],
            [17, 18, 19]
        ])
        # fmt: on
        r = input_level(X, Y, "pearson")
        self.assertAlmostEqual(r, 0.05535747748, places=4)

    def test_input_level_nan_values(self):
        # This will skip the NaN entry from the first column
        # fmt: off
        X = np.array([
            [1, 9, 2],
            [np.nan, 5, 4],
            [6, 7, 7]
        ])
        Y = np.array([
            [11, 12, 13],
            [np.nan, 15, 16],
            [17, 18, 19]
        ])
        # fmt: on
        r = input_level(X, Y, "pearson")
        self.assertAlmostEqual(r, 0.4977997559, places=4)

    @pytest.mark.filterwarnings("ignore: An input array is constant")
    def test_input_level_skipped_column(self):
        # The last column of x is all identical, so the
        # correlation for that column is NaN and will be skipped
        # in the overall mean
        # fmt: off
        X = np.array([
            [1, 9, 2],
            [4, 5, 2],
            [6, 7, 2]
        ])
        Y = np.array([
            [11, 12, 13],
            [14, 15, 16],
            [17, 18, 19]
        ])
        # fmt: on
        r = input_level(X, Y, "pearson")
        self.assertAlmostEqual(r, 0.2466996339, places=4)

    @pytest.mark.filterwarnings("ignore: An input array is constant")
    def test_input_level_nan_correlations(self):
        # Both columns of X are identical, so their correlations
        # will be NaN and the overall correlation will be NaN
        # fmt: off
        X = np.array([
            [1, 2],
            [1, 2],
        ])
        Y = np.array([
            [11, 12],
            [14, 15],
        ])
        # fmt: on
        r = input_level(X, Y, "pearson")
        assert np.isnan(r)

    def test_input_level_iv(self):
        # One dimensional inputs
        # fmt: off
        X = np.array(
            [1, 9, 2]
        )
        Y = np.array([
            [11, 12, 13],
            [14, 15, np.nan],
            [17, np.nan, 19]
        ])
        # fmt: on
        message = "`X` must be two-dimensional"
        with pytest.raises(ValueError, match=message):
            input_level(X, Y, "pearson")

        message = "`Z` must be two-dimensional"
        with pytest.raises(ValueError, match=message):
            input_level(Y, X, "pearson")

        # Different number of rows
        # fmt: off
        X = np.array([
            [1, 9, 2],
            [8, 2, 3],
        ])
        Y = np.array([
            [11, 12, 13],
            [14, 15, np.nan],
            [17, np.nan, 19]
        ])
        # fmt: on
        message = "`X` and `Z` must have the same number of rows"
        with pytest.raises(ValueError, match=message):
            input_level(Y, X, "pearson")

        # Different number of columns
        # fmt: off
        X = np.array([
            [9, 2],
            [2, 3],
            [9, 4],
        ])
        Y = np.array([
            [11, 12, 13],
            [14, 15, np.nan],
            [17, np.nan, 19]
        ])
        # fmt: on
        message = "`X` and `Z` must have the same number of columns"
        with pytest.raises(ValueError, match=message):
            input_level(Y, X, "pearson")

        # Column of NaN
        # fmt: off
        X = np.array([
            [1, 9, np.nan],
            [4, 5, np.nan],
            [6, 7, np.nan]
        ])
        Y = np.array([
            [11, 12, 13],
            [14, 15, 16],
            [17, 18, 19]
        ])
        # fmt: on
        message = "`X` must not have a column of all NaN"
        with pytest.raises(ValueError, match=message):
            input_level(X, Y, "pearson")

        message = "`Z` must not have a column of all NaN"
        with pytest.raises(ValueError, match=message):
            input_level(Y, X, "pearson")

        # Different NaN locations
        # fmt: off
        X = np.array([
            [1, 9, np.nan],
            [4, 5, 4],
            [6, 7, 2]
        ])
        Y = np.array([
            [11, 12, 13],
            [np.nan, 15, 16],
            [17, 18, 19]
        ])
        # fmt: on
        message = "`X` and `Z` must have identical NaN locations"
        with pytest.raises(ValueError, match=message):
            input_level(X, Y, "pearson")

    def test_global_level(self):
        # fmt: off
        X = np.array([
            [1, 9, 2],
            [4, 5, 2],
            [6, 7, 2]
        ])
        Y = np.array([
            [11, 12, 13],
            [14, 15, 16],
            [17, 18, 19]
        ])
        # fmt: on
        r = global_level(X, Y, "pearson")
        self.assertAlmostEqual(r, 0.06691496051, places=4)

    def test_global_level_nans(self):
        # fmt: off
        X = np.array([
            [1, 9, 2],
            [4, 5, np.nan],
            [np.nan, 7, 2]
        ])
        Y = np.array([
            [11, 12, 13],
            [14, 15, np.nan],
            [np.nan, 18, 19]
        ])
        # fmt: on
        r = global_level(X, Y, "pearson")
        self.assertAlmostEqual(r, 0.05436067275, places=4)

        # Almost all NaN
        X = np.array(
            [[1, np.nan, np.nan], [4, np.nan, np.nan], [np.nan, np.nan, np.nan]]
        )
        Y = np.array(
            [[11, np.nan, np.nan], [4, np.nan, np.nan], [np.nan, np.nan, np.nan]]
        )
        # fmt: on
        r = global_level(X, Y, "pearson")
        self.assertAlmostEqual(r, -1, places=4)

    def test_global_level_iv(self):
        # One dimensional inputs
        # fmt: off
        X = np.array(
            [1, 9, 2]
        )
        Y = np.array([
            [11, 12, 13],
            [14, 15, np.nan],
            [17, np.nan, 19]
        ])
        # fmt: on
        message = "`X` must be two-dimensional"
        with pytest.raises(ValueError, match=message):
            global_level(X, Y, "pearson")

        message = "`Z` must be two-dimensional"
        with pytest.raises(ValueError, match=message):
            global_level(Y, X, "pearson")

        # Different number of rows
        # fmt: off
        X = np.array([
            [1, 9, 2],
            [8, 2, 3],
        ])
        Y = np.array([
            [11, 12, 13],
            [14, 15, np.nan],
            [17, np.nan, 19]
        ])
        # fmt: on
        message = "`X` and `Z` must have the same number of rows"
        with pytest.raises(ValueError, match=message):
            global_level(Y, X, "pearson")

        # Different number of columns
        # fmt: off
        X = np.array([
            [9, 2],
            [2, 3],
            [9, 4],
        ])
        Y = np.array([
            [11, 12, 13],
            [14, 15, np.nan],
            [17, np.nan, 19]
        ])
        # fmt: on
        message = "`X` and `Z` must have the same number of columns"
        with pytest.raises(ValueError, match=message):
            global_level(Y, X, "pearson")

        # Different NaN locations
        # fmt: off
        X = np.array([
            [1, 9, np.nan],
            [4, 5, 4],
            [6, 7, 2]
        ])
        Y = np.array([
            [11, 12, 13],
            [np.nan, 15, 16],
            [17, 18, 19]
        ])
        # fmt: on
        message = "`X` and `Z` must have identical NaN locations"
        with pytest.raises(ValueError, match=message):
            global_level(X, Y, "pearson")

        # Not enough non-NaN values
        # fmt: off
        X = np.array([
            [1, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan]
        ])
        Y = np.array([
            [11, 12, 13],
            [np.nan, 15, 16],
            [17, 18, 19]
        ])
        # fmt: on
        message = "`X` must have at least 2 non-NaN values"
        with pytest.raises(ValueError, match=message):
            global_level(X, Y, "pearson")

        message = "`Z` must have at least 2 non-NaN values"
        with pytest.raises(ValueError, match=message):
            global_level(Y, X, "pearson")

    def test_correlate(self):
        # fmt: off
        X = np.array([
            [1, 9, 2],
            [4, 5, 2],
            [6, 7, 1]
        ])
        Y = np.array([
            [11, 12, 13],
            [14, 15, 16],
            [17, 18, 19]
        ])
        # fmt: on
        r = correlate(X, Y, "system", "pearson")
        self.assertAlmostEqual(r, 0.6546536707, places=4)

        r = correlate(X, Y, "input", "pearson")
        self.assertAlmostEqual(r, -0.124208712, places=4)

        r = correlate(X, Y, "global", "pearson")
        self.assertAlmostEqual(r, 0.0, places=4)

        r = correlate(X, Y, "system", "spearman")
        self.assertAlmostEqual(r, 0.5, places=4)

        r = correlate(X, Y, "input", "spearman")
        self.assertAlmostEqual(r, -0.1220084679, places=4)

        r = correlate(X, Y, "global", "spearman")
        self.assertAlmostEqual(r, 0.04201829034, places=4)

        r = correlate(X, Y, "system", "kendall")
        self.assertAlmostEqual(r, 0.33333333333, places=4)

        r = correlate(X, Y, "input", "kendall")
        self.assertAlmostEqual(r, -0.0499433047, places=4)

        r = correlate(X, Y, "global", "kendall")
        self.assertAlmostEqual(r, 0.11433239009, places=4)

    def test_correlate_iv(self):
        # fmt: off
        X = np.array([
            [1, 9, 2],
            [4, 5, 2],
            [6, 7, 1]
        ])
        Y = np.array([
            [11, 12, 13],
            [14, 15, 16],
            [17, 18, 19]
        ])
        # fmt: on
        message = "Unknown correlation level"
        with pytest.raises(ValueError, match=message):
            correlate(X, Y, "UNK", "kendall")
