from typing import List, Tuple
import numpy as np

from src.data.loader import MachineData


def make_windows(
    series: np.ndarray,
    labels: np.ndarray,
    W: int,
    H: int,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding-window samples from a single (multivariate) time series

    Args:
        series: time series
        labels: incident labels (0 or 1)
        W: look-back window length
        H: how many steps ahead to predict
        step: stride between consecutive windows

    Returns:
        X: windows
        y: incident labels
        t: idx of the last timestep in each window
    """
    if series.ndim == 1:
        series = series.reshape(-1, 1)
    T, F = series.shape

    if W + H > T:
        raise ValueError(f"W + H = {W + H} exceeds series length {T}. ")

    end_indices = np.arange(W - 1, T - H, step, dtype=np.int32)
    N = len(end_indices)

    X = np.empty((N, W, F), dtype=np.float32)
    y = np.empty(N, dtype=np.int8)

    for i, t in enumerate(end_indices):
        X[i] = series[t - W + 1 : t + 1]
        y[i] = int(labels[t + 1 : t + H + 1].any())

    return X, y, end_indices


def make_windows_for_machines(
    machine_list: List[MachineData],
    W: int,
    H: int,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates and concatenates windows from a list of machines, each machine's full series are
    windowed independently and then all windows are stacked

    Args:
        machine_list: list ofMachineData objects.
        W: look-back window length
        H: how many steps ahead to predict
        step: stride between consecutive windows
    """
    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_ids: List[np.ndarray] = []

    for idx, machine in enumerate(machine_list):
        X, y, _ = make_windows(machine.series, machine.labels, W, H, step)
        all_X.append(X)
        all_y.append(y)
        all_ids.append(np.full(len(y), idx, dtype=np.int32))

    return (
        np.concatenate(all_X, axis=0),
        np.concatenate(all_y, axis=0),
        np.concatenate(all_ids, axis=0),
    )
