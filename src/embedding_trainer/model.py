"""Base model loading."""

import torch
from sentence_transformers import SentenceTransformer


def load_model(model_name: str, device: torch.device, max_seq_length: int | None = None) -> SentenceTransformer:
    """Load a sentence-transformers model onto the given device.

    Forces float32 regardless of the checkpoint's native dtype (some models,
    e.g. Qwen3-Embedding, ship as bfloat16, which is numerically unstable
    for full-parameter fine-tuning on the MPS backend and can produce NaN
    losses). Optionally caps max_seq_length, since a model's own default can
    be far longer than needed for sentence embeddings and make a few long
    inputs blow up per-step compute quadratically.
    """
    model = SentenceTransformer(model_name, device=str(device), model_kwargs={"torch_dtype": torch.float32})
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
    return model
