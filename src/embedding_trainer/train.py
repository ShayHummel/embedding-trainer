"""Fine-tuning pipeline built on SentenceTransformerTrainer."""

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import ContrastiveLoss, CosineSimilarityLoss

from datasets import Dataset

from embedding_trainer.config import TrainConfig


def build_evaluator(eval_dataset: Dataset, name: str = "eval") -> EmbeddingSimilarityEvaluator:
    return EmbeddingSimilarityEvaluator(
        sentences1=eval_dataset["sentence1"],
        sentences2=eval_dataset["sentence2"],
        scores=eval_dataset["score"],
        name=name,
    )


def build_trainer(
    model: SentenceTransformer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    is_binary_score: bool,
    config: TrainConfig,
) -> SentenceTransformerTrainer:
    loss = ContrastiveLoss(model) if is_binary_score else CosineSimilarityLoss(model)
    evaluator = build_evaluator(eval_dataset)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to="tensorboard",
        logging_dir=str(config.output_dir / "logs"),
        seed=config.seed,
    )

    return SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
