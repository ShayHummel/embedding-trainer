"""Device selection for Apple Silicon (MPS) with CPU fallback."""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch


def get_device() -> torch.device:
    """Return the MPS device when available, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
