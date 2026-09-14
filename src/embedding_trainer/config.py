"""Training configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    data_csv: Path = Path("data/train.csv")
    output_dir: Path = Path("runs/embedding-trainer")
    num_train_epochs: float = 3.0
    train_batch_size: int = 32
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    eval_ratio: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        self.data_csv = Path(self.data_csv)
        self.output_dir = Path(self.output_dir)
