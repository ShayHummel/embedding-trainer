"""Base model loading."""

import torch
from sentence_transformers import SentenceTransformer


def load_model(model_name: str, device: torch.device) -> SentenceTransformer:
    """Load a sentence-transformers model onto the given device."""
    return SentenceTransformer(model_name, device=str(device))
