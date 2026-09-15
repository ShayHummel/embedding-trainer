"""CLI entry point: train, evaluate, and demo a sentence embedding model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from embedding_trainer.config import TrainConfig
from embedding_trainer.data import load_dataset_for_training
from embedding_trainer.device import get_device
from embedding_trainer.model import load_model
from embedding_trainer.train import build_trainer
from embedding_trainer.visualize import show_vector_before_after


def cmd_train(args: argparse.Namespace) -> None:
    config = TrainConfig(
        base_model=args.base_model,
        data_csv=args.data,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        max_rows=args.max_rows,
        english_only=not args.no_english_filter,
        oversample_factor=args.oversample_factor,
    )

    device = get_device()
    print(f"Using device: {device}")

    train_dataset, eval_dataset, training_format = load_dataset_for_training(
        config.data_csv,
        eval_ratio=config.eval_ratio,
        seed=config.seed,
        max_rows=config.max_rows,
        english_only=config.english_only,
        oversample_factor=config.oversample_factor,
    )
    print(f"Train examples: {len(train_dataset)} | Eval examples: {len(eval_dataset)}")
    print(f"Training format: {training_format}")

    base_model = load_model(config.base_model, device)

    trainer = build_trainer(base_model, train_dataset, eval_dataset, training_format, config)
    trainer.train()

    final_model_dir = config.output_dir / "final"
    trainer.model.save_pretrained(str(final_model_dir))
    print(f"Saved fine-tuned model to {final_model_dir}")

    if args.demo_text:
        base_model_for_demo = load_model(config.base_model, device)
        fine_tuned_model = load_model(str(final_model_dir), device)
        show_vector_before_after(
            args.demo_text,
            base_model_for_demo,
            fine_tuned_model,
            plot_path=config.output_dir / "vector_before_after.png",
        )


def cmd_demo(args: argparse.Namespace) -> None:
    device = get_device()
    print(f"Using device: {device}")

    base_model = load_model(args.base_model, device)
    fine_tuned_model = load_model(args.fine_tuned_model, device)
    show_vector_before_after(
        args.text,
        base_model,
        fine_tuned_model,
        plot_path=Path(args.plot_path) if args.plot_path else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train, test, and validate a sentence embedding model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Fine-tune the base embedding model on a CSV dataset.")
    train_parser.add_argument("--data", type=Path, default=Path("data/train.csv"), help="Path to the training CSV.")
    train_parser.add_argument(
        "--base-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Base model name or path."
    )
    train_parser.add_argument("--output-dir", type=Path, default=Path("runs/embedding-trainer"))
    train_parser.add_argument("--epochs", type=float, default=3.0)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--eval-ratio", type=float, default=0.1)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument(
        "--max-rows",
        type=int,
        default=100_000,
        help="Max rows to sample for the unsupervised (label-free text pool) schema.",
    )
    train_parser.add_argument(
        "--no-english-filter",
        action="store_true",
        help="Disable the English-only language filter for the unsupervised schema.",
    )
    train_parser.add_argument(
        "--oversample-factor",
        type=int,
        default=3,
        help="For the unsupervised schema: how many extra rows to sample before "
        "language-filtering down to --max-rows.",
    )
    train_parser.add_argument(
        "--demo-text", type=str, default=None, help="If set, run show_vector_before_after on this text after training."
    )
    train_parser.set_defaults(func=cmd_train)

    demo_parser = subparsers.add_parser(
        "demo", help="Compare base vs. fine-tuned embeddings for a single input text."
    )
    demo_parser.add_argument("text", type=str, help="Input text to encode with both models.")
    demo_parser.add_argument(
        "--base-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Base model name or path."
    )
    demo_parser.add_argument(
        "--fine-tuned-model", type=str, required=True, help="Path to the fine-tuned model directory."
    )
    demo_parser.add_argument(
        "--plot-path", type=str, default="runs/vector_before_after.png", help="Where to save the comparison plot."
    )
    demo_parser.set_defaults(func=cmd_demo)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
