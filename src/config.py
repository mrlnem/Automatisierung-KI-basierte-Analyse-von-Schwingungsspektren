from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    fs: int = 50_000
    n_fft: int = 4096
    hop_length: int = 1024
    fmax: int = 20_000

    img_size: int = 256
    db_min: float = -80.0
    db_max: float = 0.0

    train_cutters = ("c1", "c4", "c6")
    test_cutters = ("c2", "c3", "c5")

    classes = ("uniform", "severe", "rapid")

    results_dir: Path = Path("results")
