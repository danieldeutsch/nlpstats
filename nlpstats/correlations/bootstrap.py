import numpy as np
import numpy.typing as npt
from collections import namedtuple
from typing import Callable, Union

from nlpstats.correlations.correlations import correlate
from nlpstats.correlations.resampling import resample

BootstrapResult = namedtuple("BootstrapResult", ["lower", "upper", "samples"])


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
    X: npt.ArrayLike, Z: npt.ArrayLike, level: str, paired_inputs: bool, confidence_level: float, n_resamples: int,
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
