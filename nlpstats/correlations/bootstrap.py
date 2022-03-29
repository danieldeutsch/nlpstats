import numpy as np
import numpy.typing as npt
from typing import Callable, NamedTuple, List, Union

from nlpstats.correlations.correlations import correlate
from nlpstats.correlations.resampling import resample


class BootstrapResult(NamedTuple):
    lower: float
    """The lower-bound"""

    upper: float
    """The upper-bound"""

    samples: List[float]
    """The bootstrapped samples"""


def bootstrap(
    X: npt.ArrayLike,
    Z: npt.ArrayLike,
    level: str,
    coefficient: Union[Callable, str],
    resampling_method: Union[Callable, str],
    paired_inputs: bool = True,
    confidence_level: float = 0.95,
    n_resamples: int = 9999,
) -> BootstrapResult:
    """Calculates a confidence interval for a correlation via bootstrapping.

    The rows of :code:`X` and :code:`Z` should always correspond to each other.
    That is, :code:`X[i]` and :code:`Z[i]` contain the scores for the outputs from
    system :code:`i`. If the columns of :code:`X` and :code:`Z` correspond to each
    other, :code:`paired_inputs` should be set to :code:`True`, and this is required
    for input- and global-level correlations.

    If a score is missing for a specific output, that value should be equal to
    :code:`np.nan`. For input- and global-level correlations, :code:`X` and
    :code:`Z` must have :code:`np.nan` values in the same locations.

    The resampling method indicates whether the set of systems (rows) and/or
    inputs (columns) are resampled during bootstrapping. They correspond to the
    Boot-Inputs, Boot-Systems, and Boot-Both resampling methods from
    `Deutsch et al. (2021) <https://arxiv.org/abs/2104.00054>`_. Each method
    results in different interpretations for the resulting confidence intervals.

    Parameters
    ----------
    X : npt.ArrayLike
        A two-dimensional score matrix in which :code:`X[i][j]` contains the
        :code:`X` score for the :code:`i` th system on the :code:`j` th input.
    Z : npt.ArrayLike
        A two-dimensional score matrix in which :code:`Z[i][j]` contains the
        :code:`Z` score for the :code:`i` th system on the :code:`j` th input.
    level : str
        The level of correlation, either :code:`"system"`, :code:`"input"`, or :code:`"global"`.
    coefficient : Union[Callable, str]
        The correlation coefficient to use, either :code:`"pearson"`, :code:`"spearman"`,
        :code:`"kendall"`, or a custom correlation function. The custom function
        must accept two vectors as input and return the correlation between them.
    resampling_method : str
        The resampling method to use, either :code:`"systems"`, :code:`"inputs"`, or :code:`"both"`
        to indicate whether the systems and/or inputs should be resampled during bootstrapping.
        If :code:`paired_inputs=True` and inputs are resampled, they will be sampled in
        parallel for :code:`X` and :code:`Z`, otherwise they will not.
    paired_inputs : bool
        Indicates whether the columns of :code:`X` and :code:`Z` are paired
    confidence_level : float
        The confidence level of the correlation interval, between 0 and 1.
    n_resamples : int
        The number of resamples to take

    Returns
    -------
    BootstrapResult

    Examples
    --------
    Given score matrices :code:`X` and :code:`Z`, the 95% confidence interval for
    the system-level Pearson correlation using the Boot-Both resampling method
    can be calculated as:

    >>> import numpy as np
    >>> m, n = 10, 25
    >>> X = np.random.rand(m, n)
    >>> Z = np.random.rand(m, n)
    >>> bootstrap(X, Z, "system", "pearson", "both")
    """
    X, Z = _bootstrap_iv(X, Z, level, paired_inputs, confidence_level, n_resamples)

    samples = []
    for _ in range(n_resamples):
        X_s, Z_s = resample((X, Z), resampling_method, paired_inputs=paired_inputs)
        r = correlate(X_s, Z_s, level, coefficient)
        if not np.isnan(r):
            samples.append(r)

    alpha = (1 - confidence_level) / 2
    lower = np.percentile(samples, alpha * 100)
    upper = np.percentile(samples, (1 - alpha) * 100)
    return BootstrapResult(lower, upper, samples)


def _bootstrap_iv(
    X: npt.ArrayLike,
    Z: npt.ArrayLike,
    level: str,
    paired_inputs: bool,
    confidence_level: float,
    n_resamples: int,
):
    X = np.asarray(X)
    Z = np.asarray(Z)

    if not paired_inputs and level in {"input", "global"}:
        raise ValueError(
            f"`paired_inputs` must be `True` for input- or global-level correlations"
        )

    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError(f"`confidence_level` must be between 0 and 1 (exclusive)")

    if n_resamples <= 0:
        raise ValueError(f"`n_resamples` must be a positive integer")

    return X, Z
