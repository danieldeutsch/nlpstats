import numpy as np
import numpy.typing as npt
from typing import Callable, List, NamedTuple, Tuple, Union

from nlpstats.correlations.correlations import correlate
from nlpstats.correlations.resampling import permute


class PermutationResult(NamedTuple):
    pvalue: float
    """The p-value of the test"""

    samples: List[float]
    """The values of the test statistic sampled during the test"""


def _standardize(X: np.ndarray) -> np.ndarray:
    return (X - np.nanmean(X)) / np.nanstd(X)


def permutation_test(
    X: npt.ArrayLike,
    Y: npt.ArrayLike,
    Z: npt.ArrayLike,
    level: str,
    coefficient: Union[Callable, str],
    permutation_method: Union[Callable, str],
    alternative: str = "two-sided",
    n_resamples: int = 9999,
) -> PermutationResult:
    """Runs a hypothesis test on the difference between correlations.

    This function will compare the correlations of :code:`X` to :code:`Z`
    and :code:`Y` to :code:`Z`. Typically, :code:`X` and :code:`Y`
    correspond to metric score matrices and :code:`Z` is the human score matrix.

    The rows of :code:`X`, :code:`Y`, and :code:`Z` must correspond
    to each other, and the columns of :code:`X` and :code:`Y` must too.
    For input- and global-level correlations, the columns of :code:`X`
    and :code:`Y` must also be paired with those of :code:`Z`.

    If a value from the matrices is missing, it should be replaced with
    :code:`np.nan`. The :code:`np.nan` locations must always be identical for
    :code:`X` and :code:`Y`. The same is true for :code:`Z` for input- and
    global-level correlations.

    The permutation method indicates whether the set of systems (rows) and/or
    inputs (columns) are permuted during the test. They correspond to the
    Perm-Inputs, Perm-Systems, and Perm-Both permutation methods from
    `Deutsch et al. (2021) <https://arxiv.org/abs/2104.00054>`_. Each method
    results in different interpretations for the resulting hypothesis tests.

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
        :code:`"kendall"`, or a custom correlation function. The custom function
        must accept two vectors as input and return the correlation between them.
    permutation_method : str
        The permutation method to use, either :code:`"systems"`, :code:`"inputs"`, or :code:`"both"`
        to indicate whether the systems and/or inputs should be permuted during the test.
    alternative : str
        The alternative hypothesis. :code:`"two-sided"` corresponds to an alternative
        hypothesis that :math:`r(X, Z) \\neq r(Y, Z)`, :code:`"greater"` correponds
        to :math:`r(X, Z) > r(Y, Z)` and :code:`"less"` corresponds to
        :math:`r(X, Z) < r(Y, Z)`.
    n_resamples : int
        The number of permutation samples to take

    Returns
    -------
    PermutationResult

    Examples
    --------
    An example of hypothesis testing the input-level Pearson correlation of :code:`X`
    to :code:`Z` and :code:`Y` to :code:`Z` using Perm-Inputs.

    >>> import numpy as np
    >>> m, n = 10, 25
    >>> X = np.random.rand(m, n)
    >>> Y = np.random.rand(m, n)
    >>> Z = np.random.rand(m, n)
    >>> permutation_test(X, Y, Z, "input", "pearson", "inputs")

    The system-level correlations can be compared even if the columns of :code:`X`
    and :code:`Y` to do not match :code:`Z`:

    >>> X = np.random.rand(m, 2 * n)
    >>> Y = np.random.rand(m, 2 * n)
    >>> permutation_test(X, Y, Z, "system", "pearson", "inputs")
    """
    X, Y, Z = _permutation_test_iv(X, Y, Z, level, alternative, n_resamples)

    # The data needs to be standardized so the metrics are on the same scale.
    X = _standardize(X)
    Y = _standardize(Y)
    Z = _standardize(Z)

    observed = correlate(X, Z, level, coefficient) - correlate(Y, Z, level, coefficient)

    samples = []
    count = 0
    for _ in range(n_resamples):
        X_p, Y_p = permute(X, Y, permutation_method)
        sample = correlate(X_p, Z, level, coefficient) - correlate(
            Y_p, Z, level, coefficient
        )
        samples.append(sample)

        if alternative == "two-sided":
            if abs(sample) >= abs(observed):
                count += 1
        elif alternative == "greater":
            if sample >= observed:
                count += 1
        elif alternative == "less":
            if sample <= observed:
                count += 1
        else:
            raise ValueError(f"Unknown alternative: {alternative}")

    pvalue = count / len(samples)
    return PermutationResult(pvalue, samples)


def _permutation_test_iv(
    X: npt.ArrayLike,
    Y: npt.ArrayLike,
    Z: npt.ArrayLike,
    level: str,
    alternative: str,
    n_resamples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(f"Unknown alternative: {alternative}")

    if n_resamples <= 0:
        raise ValueError(f"`n_resamples` must be a positive integer")

    return X, Y, Z
