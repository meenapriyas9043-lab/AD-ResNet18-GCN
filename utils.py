"""
Shared utilities: config loading and reproducibility.
"""
import random
import yaml
import numpy as np
import torch


def load_config(path: str = "config.yaml") -> dict:
    """Load the single project-wide config file (see repo root)."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Fix every relevant RNG so folds, splits, and augmentations are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
