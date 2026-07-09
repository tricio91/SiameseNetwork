"""Small shared helpers."""
from __future__ import annotations

import random

import numpy as np
import torch


def get_device() -> torch.device:
    """Return the CUDA device if one is available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
