import numpy as np
import numpy.typing as npt
from scipy.stats import kendalltau, pearsonr, spearmanr
from typing import Callable, Tuple, Union


def _kendalltau(*args):
    return kendalltau(*args)[0]


def _pearsonr(*args):
    return pearsonr(*args)[0]


def _spearmanr(*args):
    return spearmanr(*args)[0]


def _coefficient_iv(coefficient: Union[Callable, str]) -> Callable:
    if isinstance(coefficient, str):
        if coefficient == "kendall":
            return _kendalltau
        elif coefficient == "pearson":
            return _pearsonr
        elif coefficient == "spearman":
            return _spearmanr
        else:
            raise ValueError(f"Unknown correlation coefficient: {coefficient}")
    return coefficient


def _correlation_iv(
    X: npt.ArrayLike, Z: npt.ArrayLike, coefficient: Union[Callable, str]
) -> Tuple[np.ndarray, np.ndarray, Callable]:
    X = np.asarray(X)
    Z = np.asarray(Z)
    if X.ndim != 2:
        raise ValueError(f"`X` must be two-dimensional")
    if Z.ndim != 2:
        raise ValueError(f"`Z` must be two-dimensional")
    if X.shape[0] != Z.shape[0]:
        raise ValueError(f"`X` and `Z` must have the same number of rows")

    coefficient = _coefficient_iv(coefficient)
    return X, Z, coefficient


def system_level(
    X: npt.ArrayLike, Z: npt.ArrayLike, coefficient: Union[Callable, str]
) -> float:
    X, Z, coefficient = _system_level_iv(X, Z, coefficient)
    x = np.nanmean(X, axis=1)
    z = np.nanmean(Z, axis=1)
    return coefficient(x, z)


def _system_level_iv(
    X: npt.ArrayLike, Z: npt.ArrayLike, coefficient: Union[Callable, str]
) -> Tuple[np.ndarray, np.ndarray, Callable]:
    X, Z, coefficient = _correlation_iv(X, Z, coefficient)

    n1 = X.shape[1]
    n2 = Z.shape[1]
    if (np.isnan(X).sum(axis=1) == n1).any():
        raise ValueError(f"`X` must not have a row of all NaN")
    if (np.isnan(Z).sum(axis=1) == n2).any():
        raise ValueError(f"`Z` must not have a row of all NaN")

    return X, Z, coefficient


def input_level(
    X: npt.ArrayLike,
    Z: npt.ArrayLike,
    coefficient: Union[Callable, str],
) -> float:
    X, Z, coefficient = _input_level_iv(X, Z, coefficient)
    n = X.shape[1]
    rs = []
    for j in range(n):
        x, z = X[:, j], Z[:, j]

        # Remove any NaN. Because the input validation checks to
        # ensure the NaNs are in identical locations, these vectors
        # will be paired
        x = x[~np.isnan(x)]
        z = z[~np.isnan(z)]
        r = coefficient(x, z)
        if not np.isnan(r):
            rs.append(r)

    if len(rs) == 0:
        return np.nan
    return np.mean(rs)


def _input_level_iv(
    X: npt.ArrayLike,
    Z: npt.ArrayLike,
    coefficient: Union[Callable, str],
) -> Tuple[np.ndarray, np.ndarray, Callable]:
    X, Z, coefficient = _correlation_iv(X, Z, coefficient)

    if X.shape[1] != Z.shape[1]:
        raise ValueError(f"`X` and `Z` must have the same number of columns")

    m1 = X.shape[0]
    m2 = Z.shape[0]
    if (np.isnan(X).sum(axis=0) == m1).any():
        raise ValueError(f"`X` must not have a column of all NaN")
    if (np.isnan(Z).sum(axis=0) == m2).any():
        raise ValueError(f"`Z` must not have a column of all NaN")

    if np.not_equal(np.isnan(X), np.isnan(Z)).any():
        raise ValueError(f"`X` and `Z` must have identical NaN locations")

    return X, Z, coefficient


def global_level(
    X: npt.ArrayLike,
    Z: npt.ArrayLike,
    coefficient: Union[Callable, str],
) -> float:
    X, Z, coefficient = _global_level_iv(X, Z, coefficient)

    # Flatten into vectors
    x, z = X.flatten(), Z.flatten()

    # Remove NaNs. x and z will still be paired because
    # the input validation checks to make sure the NaNs are in
    # identical locations
    x = x[~np.isnan(x)]
    z = z[~np.isnan(z)]

    return coefficient(x, z)


def _global_level_iv(
    X: npt.ArrayLike,
    Z: npt.ArrayLike,
    coefficient: Union[Callable, str],
) -> Tuple[np.ndarray, np.ndarray, Callable]:
    X, Z, coefficient = _correlation_iv(X, Z, coefficient)

    if X.shape[1] != Z.shape[1]:
        raise ValueError(f"`X` and `Z` must have the same number of columns")

    if X.size - np.isnan(X).sum() < 2:
        raise ValueError(f"`X` must have at least 2 non-NaN values")
    if Z.size - np.isnan(Z).sum() < 2:
        raise ValueError(f"`Z` must have at least 2 non-NaN values")

    if np.not_equal(np.isnan(X), np.isnan(Z)).any():
        raise ValueError(f"`X` and `Z` must have identical NaN locations")

    return X, Z, coefficient


def correlate(
    X: npt.ArrayLike,
    Z: npt.ArrayLike,
    level: str,
    coefficient: Union[Callable, str],
) -> float:
    X, Z, level = _correlate_iv(X, Z, level)
    return level(X, Z, coefficient)


def _correlate_iv(
    X: npt.ArrayLike, Z: npt.ArrayLike, level: str
) -> Tuple[np.ndarray, np.ndarray, Callable]:
    X = np.asarray(X)
    Z = np.asarray(Z)

    if level == "system":
        level = system_level
    elif level == "input":
        level = input_level
    elif level == "global":
        level = global_level
    else:
        raise ValueError(f"Unknown correlation level: {level}")

    return X, Z, level
