import numpy as np
import numpy.typing as npt
import scipy.stats
from collections import namedtuple

from nlpstats.correlations.correlations import correlate

FisherResult = namedtuple("FisherResult", ["lower", "upper"])


def fisher(
    X: npt.ArrayLike,
    Z: npt.ArrayLike,
    level: str,
    coefficient: str,
    confidence_level: float = 0.95,
) -> FisherResult:
    _fisher_iv(confidence_level)

    r = correlate(X, Z, level, coefficient)

    # See Bonett and Wright (200) for details
    if coefficient == "pearson":
        b, c = 3, 1
    elif coefficient == "spearman":
        b, c = 3, np.sqrt(1 + r ** 2 / 2)
    elif coefficient == "kendall":
        b, c = 4, np.sqrt(0.437)
    else:
        raise ValueError(f"Unknown correlation coefficient: {coefficient}")

    if level == "system":
        # The number of systems
        n = X.shape[0]
    elif level == "input":
        # Assume n is the summary-correlation with the largest n.
        # We find that by counting how many non-nans are in each column,
        # then taking the max
        n = (~np.isnan(X)).sum(axis=0).max()
    elif level == "global":
        # The number of non-NaN entries
        n = (~np.isnan(X)).sum()
    else:
        raise Exception(f"Unknown correlation level: {level}")

    alpha = 1 - confidence_level
    if n > b:
        z_r = np.arctanh(r)
        z = scipy.stats.norm.ppf(1.0 - alpha / 2)
        z_l = z_r - z * c / np.sqrt(n - b)
        z_u = z_r + z * c / np.sqrt(n - b)
        r_l = np.tanh(z_l)
        r_u = np.tanh(z_u)
    else:
        r_l, r_u = np.nan, np.nan
    return FisherResult(r_l, r_u)


def _fisher_iv(confidence_level: float) -> None:
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError(f"`confidence_level` must be between 0 and 1 (exclusive)")
