"""Training configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen3-Embedding-0.6B"
    data_csv: Path = Path("data/train.csv")
    output_dir: Path = Path("runs/embedding-trainer")
    num_train_epochs: float = 3.0
    # Qwen3-Embedding-0.6B is a ~600M-param decoder, much heavier than the
    # previous MiniLM default (22M params) — keep batches small by default
    # to fit laptop-class unified memory; raise if your machine has room.
    train_batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    eval_ratio: float = 0.1
    seed: int = 42
    # The base model's own default (32768 for Qwen3-Embedding) is far longer
    # than needed for sentence embeddings and lets a few long documents blow
    # up per-step compute quadratically; cap it for speed/memory.
    max_seq_length: int = 256

    # Only used for the unsupervised (label-free text pool) schema.
    max_rows: int = 100_000
    english_only: bool = True
    oversample_factor: int = 3

    def __post_init__(self) -> None:
        self.data_csv = Path(self.data_csv)
        self.output_dir = Path(self.output_dir)
