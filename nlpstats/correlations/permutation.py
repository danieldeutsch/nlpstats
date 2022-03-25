import numpy as np
import numpy.typing as npt
from collections import namedtuple
from typing import Callable, Tuple, Union

from nlpstats.correlations.correlations import correlate
from nlpstats.correlations.resampling import permute

PermutationResult = namedtuple("PermutationResult", ["pvalue", "samples"])


def _standardize(X: np.ndarray) -> np.ndarray:
    return (X - np.nanmean(X)) / np.nanstd(X)


def permutation_test(
    X: npt.ArrayLike,
    Y: npt.ArrayLike,
    Z: npt.ArrayLike,
    level: str,
    coefficient: Union[Callable, str],
    permutation_method: Union[Callable, str],
    paired_inputs: bool = True,
    alternative: str = "two-sided",
    n_resamples: int = 9999,
) -> PermutationResult:
    X, Y, Z = _permutation_test_iv(
        X, Y, Z, level, paired_inputs, alternative, n_resamples
    )

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
    paired_inputs: bool,
    alternative: str,
    n_resamples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    if not paired_inputs and level in {"input", "global"}:
        raise ValueError(
            f"`paired_inputs` must be `True` for input- or global-level correlations"
        )

    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(f"Unknown alternative: {alternative}")

    if n_resamples <= 0:
        raise ValueError(f"`n_resamples` must be a positive integer")

    return X, Y, Z
