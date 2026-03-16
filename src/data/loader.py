import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np


@dataclass
class MachineData:
    """Holds all data for a single server machine"""

    name: str
    train: np.ndarray      
    test: np.ndarray       
    test_labels: np.ndarray

    @property
    def n_features(self) -> int:
        return self.train.shape[1]

    @property
    def series(self) -> np.ndarray:
        """Full series (train then test)"""
        return np.concatenate([self.train, self.test], axis=0)

    @property
    def labels(self) -> np.ndarray:
        """
        Full label vector aligned with self.series:
          - zeros for the training
          - actual labels for the test
        """
        return np.concatenate([
            np.zeros(len(self.train), dtype=np.int8),
            self.test_labels.astype(np.int8),
        ])

    @property
    def incident_rate(self) -> float:
        """Percentage of test timesteps labeled as incident"""
        return float(self.test_labels.mean())


def _load_txt(path: Path) -> np.ndarray:
    """Load .txt file and return a numpy array"""
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def load_smd(
    raw_dir: str | Path,
    machines: Optional[List[str]] = None,
) -> List[MachineData]:
    """
    Load SMD machines

    Args:
        raw_dir:  Path containing train/, test/, test_label/
        machines: Optional list of machine names to load (loads all if None)

    Returns:
        Sorted list of MachineData objects
    """
    raw_dir = Path(raw_dir)
    train_dir  = raw_dir / "train"
    test_dir   = raw_dir / "test"
    label_dir  = raw_dir / "test_label"
    
    
    # check for missing directories/files(machines)
    for d in (train_dir, test_dir, label_dir):
        if not d.exists():
            raise FileNotFoundError(f"Expected directory not found: {d}\n\n")

    all_names = sorted(
        p.stem
        for p in train_dir.glob("*.txt")
        if re.match(r"machine-\d+-\d+", p.stem)
    )
    if not all_names:
        raise FileNotFoundError(
            f"No machine-*.txt files found in {train_dir}"
        )

    if machines is not None:
        missing = set(machines) - set(all_names)
        if missing:
            raise ValueError(f"Some machines not found in dataset: {missing}")
        all_names = [n for n in all_names if n in machines]

    
    # create MachineData object for each machine
    result: List[MachineData] = []
    for name in all_names:
        train = _load_txt(train_dir / f"{name}.txt")
        test = _load_txt(test_dir / f"{name}.txt")
        labels_raw = _load_txt(label_dir / f"{name}.txt").squeeze()

        md = MachineData(name=name, train=train, test=test, test_labels=labels_raw)
        result.append(md)

    return result


def split_machines(
    machines: List[MachineData],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[List[MachineData], List[MachineData], List[MachineData]]:
    """
    Partition machine list into train / val / test groups (in contest of machine learning)
    Split is machine level (not window level) so we can prevent data leakage
    """
    rng = np.random.default_rng(seed)

    # machines without incidents are plaqced in train
    has_incidents = [m for m in machines if m.test_labels.sum() > 0]
    no_incidents = [m for m in machines if m.test_labels.sum() == 0]

    rng.shuffle(has_incidents)

    n = len(has_incidents)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))

    test_ms  = has_incidents[:n_test]
    val_ms   = has_incidents[n_test : n_test + n_val]
    train_ms = has_incidents[n_test + n_val :] + no_incidents

    return train_ms, val_ms, test_ms
