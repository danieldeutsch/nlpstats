import numpy as np
import numpy.typing as npt
import scipy.stats
from typing import Callable, NamedTuple, Union

from nlpstats.correlations.correlations import correlate


class WilliamsResult(NamedTuple):
    pvalue: float
    """The p-value of the test"""


def williams_test(
    X: npt.ArrayLike,
    Y: npt.ArrayLike,
    Z: npt.ArrayLike,
    level: str,
    coefficient: Union[Callable, str],
    alternative: str = "two-sided",
) -> WilliamsResult:
    """Calculates a hypothesis test between two correlations using Williams' test.

    See `Graham & Baldwin (2014) <https://aclanthology.org/D14-1020.pdf>`_
    for details on Williams' test.

    The rows of :code:`X`, :code:`Y`, and :code:`Z` must correspond
    to each other, and the columns of :code:`X` and :code:`Y` must too.
    For input- and global-level correlations, the columns of :code:`X`
    and :code:`Y` must also be paired with those of :code:`Z`.

    If a value from the matrices is missing, it should be replaced with
    :code:`np.nan`. The :code:`np.nan` locations must always be identical for
    :code:`X` and :code:`Y`. The same is true for :code:`Z` for input- and
    global-level correlations.

    Parameters
    ----------
    X : npt.ArrayLike
        A two-dimensional score matrix in which :code:`X[i][j]` contains the
        :code:`X` score for the :code:`i` th system on the :code:`j` th input.
    Y : npt.ArrayLike
        A two-dimensional score matrix in which :code:`Y[i][j]` contains the
        :code:`Y` score for the :code:`i` th system on the :code:`j` th input.
    Z : npt.ArrayLike
        A two-dimensional score matrix in which :code:`Z[i][j]` contains the
        :code:`Z` score for the :code:`i` th system on the :code:`j` th input.
    level : str
        The level of correlation, either :code:`"system"`, :code:`"input"`, or :code:`"global"`.
    coefficient : Union[Callable, str]
        The correlation coefficient to use, either :code:`"pearson"`, :code:`"spearman"`,
        :code:`"kendall"`.
    alternative : str
        The alternative hypothesis. :code:`"two-sided"` corresponds to an alternative
        hypothesis that :math:`r(X, Z) \\neq r(Y, Z)`, :code:`"greater"` correponds
        to :math:`r(X, Z) > r(Y, Z)` and :code:`"less"` corresponds to
        :math:`r(X, Z) < r(Y, Z)`.

    Returns
    -------
    WilliamsResult
    """

    # In the math, Z is metric 1. We take the absolute value of the correlations because
    # it does not matter whether they are positively or negatively correlated with each other. The WMT scripts
    # do the same before calling r.test
    r12 = abs(correlate(X, Z, level, coefficient))
    r13 = abs(correlate(Y, Z, level, coefficient))
    r23 = abs(correlate(X, Y, level, coefficient))

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

    # Implementation based on https://github.com/cran/psych/blob/master/R/r.test.R
    diff = r12 - r13
    det = 1 - (r12 ** 2) - (r23 ** 2) - (r13 ** 2) + (2 * r12 * r23 * r13)
    av = (r12 + r13) / 2
    cube = (1 - r23) ** 3
    t2 = diff * np.sqrt(
        (n - 1) * (1 + r23) / ((2 * (n - 1) / (n - 3)) * det + av ** 2 * cube)
    )

    # r.test implicitly assumes that r12 > r13 because it takes the absolute value of the
    # t statistic. Since we don't, we have to have special handling for one-tailed tests
    # so we don't map a negative t statistic to a positive one.
    if alternative == "two-sided":
        pvalue = scipy.stats.t.sf(abs(t2), n - 3) * 2
    elif alternative == "greater":
        pvalue = scipy.stats.t.sf(t2, n - 3)
    elif alternative == "less":
        pvalue = scipy.stats.t.sf(-1 * t2, n - 3)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return WilliamsResult(pvalue)
