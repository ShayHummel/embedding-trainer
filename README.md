# Embedding Model Trainer

Train, test, and validate a sentence embedding model end-to-end, locally on
Apple Silicon (MPS with CPU fallback). Fine-tunes an open-weight
sentence-transformers model (default:
`sentence-transformers/all-MiniLM-L6-v2`) on your own CSV data using
`SentenceTransformerTrainer`, with evaluation metrics logged to TensorBoard.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Data

Drop a CSV into `data/`. Two schemas are auto-detected:

**1. Pairs** — `sentence1`, `sentence2`, and a `score`/`similarity` (0-1 or
any numeric range, e.g. STS-style 0-5) or `label` column:

```csv
sentence1,sentence2,score
A man is playing guitar.,A person is playing a musical instrument.,0.9
A dog is running in the park.,A cat is sleeping on the couch.,0.1
```

**2. Text + label** — a `text` column and a categorical `label` column.
Positive (same-label) and negative (different-label) pairs are sampled
automatically to build a pair dataset:

```csv
text,label
A man is playing guitar.,music
A woman is playing the piano.,music
A dog is running in the park.,animals
```

Column names are matched case-insensitively; `text1`/`text2` is also
accepted as an alias for `sentence1`/`sentence2`.

## Training

```bash
uv run main.py train --data data/train.csv
```

Useful flags: `--base-model`, `--output-dir` (defaults to
`runs/embedding-trainer`, watch it with `tensorboard --logdir runs`),
`--epochs`, `--batch-size`, `--learning-rate`, `--eval-ratio`, `--seed`, and
`--demo-text "..."` to run the before/after comparison right after training.

The fine-tuned model is saved to `<output-dir>/final`.

## Before/after comparison

`show_vector_before_after()` (`src/embedding_trainer/visualize.py`) encodes
the same input with the base model and the fine-tuned model, prints both
vectors plus their cosine similarity, and saves a comparison plot.

```bash
uv run main.py demo "Your input sentence." --fine-tuned-model runs/embedding-trainer/final
```

## Project layout

```
main.py                        CLI entry point (train / demo)
src/embedding_trainer/
  config.py                    TrainConfig dataclass
  device.py                    MPS/CPU device selection
  data.py                      CSV loading + schema auto-detection
  model.py                     Base model loading
  train.py                     SentenceTransformerTrainer setup
  visualize.py                 show_vector_before_after()
data/                          Your CSV files (gitignored)
runs/                          TensorBoard logs + saved models (gitignored)
```
