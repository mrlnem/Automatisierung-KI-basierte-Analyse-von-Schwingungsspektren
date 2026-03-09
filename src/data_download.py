import kagglehub
from pathlib import Path

def download_phm2010() -> Path:
    """
    Downloads PHM 2010 dataset via kagglehub and returns the local path.
    """
    path = kagglehub.dataset_download("rabahba/phm-data-challenge-2010")
    return Path(path)
