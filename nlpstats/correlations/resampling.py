import numpy as np
import numpy.typing as npt
from typing import Callable, List, Sequence, Tuple, Union


def _resample_systems(matrices: List[np.ndarray], **kwargs) -> List[np.ndarray]:
    _resample_systems_iv(matrices)
    m = matrices[0].shape[0]
    rows = np.random.choice(m, m, replace=True)
    return [matrix[rows] for matrix in matrices]


def _resample_systems_iv(matrices: List[np.ndarray]) -> None:
    m = matrices[0].shape[0]
    for matrix in matrices:
        if matrix.shape[0] != m:
            raise ValueError("Input `matrices` all must have the same number of rows")


def _resample_inputs(
    matrices: List[np.ndarray], paired_inputs: bool
) -> List[np.ndarray]:
    _resample_inputs_iv(matrices, paired_inputs)
    if paired_inputs:
        n = matrices[0].shape[1]
        cols = np.random.choice(n, n, replace=True)
        return [matrix[:, cols] for matrix in matrices]
    else:
        resamples = []
        for matrix in matrices:
            n = matrix.shape[1]
            cols = np.random.choice(n, n, replace=True)
            resamples.append(matrix[:, cols])
        return resamples


def _resample_inputs_iv(matrices: List[np.ndarray], paired_inputs: bool) -> None:
    if paired_inputs:
        n = matrices[0].shape[1]
        for matrix in matrices:
            if matrix.shape[1] != n:
                raise ValueError(
                    "Input `matrices` all must have the same number of columns"
                )


def _resample_both(matrices: List[np.ndarray], paired_inputs: bool) -> List[np.ndarray]:
    _resample_both_iv(matrices, paired_inputs)
    matrices = _resample_systems(matrices)
    return _resample_inputs(matrices, paired_inputs)


def _resample_both_iv(matrices: List[np.ndarray], paired_inputs: bool) -> None:
    _resample_systems_iv(matrices)
    _resample_inputs_iv(matrices, paired_inputs)


def resample(
    matrices: Union[npt.ArrayLike, Sequence[npt.ArrayLike]],
    resampling_method: Union[Callable, str],
    paired_inputs: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    matrices, resampling_method = _resample_iv(matrices, resampling_method)
    resamples = resampling_method(matrices, paired_inputs=paired_inputs)
    if len(resamples) == 1:
        return resamples[0]
    return tuple(resamples)


def _resample_iv(
    matrices: Union[npt.ArrayLike, Sequence[npt.ArrayLike]],
    resampling_method: Union[Callable, str],
) -> Tuple[Sequence[np.ndarray], Callable]:
    # If the input is just one matrix, wrap it into a list
    if isinstance(matrices, np.ndarray):
        matrices = [matrices]

    # Ensure all are numpy arrays
    for matrix in matrices:
        if not isinstance(matrix, np.ndarray):
            raise TypeError(f"Input `matrices` must all be of type `np.ndarray`")

    if isinstance(resampling_method, str):
        if resampling_method == "systems":
            resampling_method = _resample_systems
        elif resampling_method == "inputs":
            resampling_method = _resample_inputs
        elif resampling_method == "both":
            resampling_method = _resample_both
        else:
            raise ValueError(f"Unknown resampling method: {resampling_method}")

    return matrices, resampling_method
