import numpy as np
import numpy.typing as npt
import scipy.stats
from typing import NamedTuple

from nlpstats.correlations.correlations import correlate


class FisherResult(NamedTuple):
    lower: float
    """The lower-bound"""

    upper: float
    """The upper-bound"""


def fisher(
    X: npt.ArrayLike,
    Z: npt.ArrayLike,
    level: str,
    coefficient: str,
    confidence_level: float = 0.95,
) -> FisherResult:
    """Calculates a confidence interval for a correlation via the Fisher transformation.

    The Fisher function is a parametric method for calculating the confidence interval
    for a correlation (see `Bonett & Wright (2000) <https://link.springer.com/content/pdf/10.1007/BF02294183.pdf>`_).

    The rows of :code:`X` and :code:`Z` should always correspond to each other.
    That is, :code:`X[i]` and :code:`Z[i]` contain the scores for the outputs from
    system :code:`i`. For input- and global-level correlations, the columns
    should also correspond to each other and thus :code:`X` and :code:`Z` must be
    the same shape; there is no such requirement for system-level correlations.

    If a score is missing for a specific output, that value should be equal to
    :code:`np.nan`. For input- and global-level correlations, :code:`X` and
    :code:`Z` must have :code:`np.nan` values in the same locations.

    Parameters
    ----------
    X : npt.ArrayLike
        A two-dimensional score matrix
    Z : npt.ArrayLike
        A two-dimensional score matrix
    level : str
        The level of correlation, either :code:`"system"`, :code:`"input"`, or :code:`"global"`.
    coefficient : Union[Callable, str]
        The correlation coefficient to use, either :code:`"pearson"`, :code:`"spearman"`,
        :code:`"kendall"`.
    confidence_level : float
        The confidence level of the correlation interval, between 0 and 1.

    Returns
    -------
    FisherResult
    """
    _fisher_iv(confidence_level)

    r = correlate(X, Z, level, coefficient)

    # See Bonett and Wright (2000) for details
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
