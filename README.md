# Embedding Model Trainer

Train, test, and validate a sentence embedding model end-to-end, locally on
Apple Silicon (MPS with CPU fallback). Fine-tunes an open-weight
sentence-transformers model (default: `Qwen/Qwen3-Embedding-0.6B`) on your
own CSV data using `SentenceTransformerTrainer`, with evaluation metrics
logged to TensorBoard.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Data

Drop a CSV into `data/`. Three schemas are auto-detected:

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

**3. Unsupervised text pool** — a single free-text column (`text`,
`content`, or `sentence`) with **no** label/score, e.g. a large unlabeled
scrape:

```csv
text_id,content,source_url
TXT_1,Some article text...,https://example.com/a
TXT_2,Some other article text...,https://example.com/b
```

Used when no pair/label columns are found. Since there's nothing to
supervise on, this trains with a SimCSE-style objective: each sampled
sentence is duplicated as its own positive pair and relies on dropout noise
across two forward passes to differ, with in-batch negatives
(`MultipleNegativesRankingLoss`). Because files like this can be huge, the
loader reservoir-samples a bounded number of rows in a single streaming
pass rather than loading the whole CSV, and by default keeps only rows
detected as English (project scope is English-only) — see
`--max-rows`, `--no-english-filter`, `--oversample-factor` below. A
held-out slice is turned into (sentence1, sentence2, score) pairs — each
text against itself (score 1.0) and against a random other text (score
0.0) — purely for evaluation, so correlation metrics still show up in
TensorBoard the same as for the other two schemas.

## Training

```bash
uv run main.py train --data data/train.csv
```

Useful flags: `--base-model`, `--output-dir` (defaults to
`runs/embedding-trainer`, watch it with `tensorboard --logdir runs`),
`--epochs`, `--batch-size` (default 8 — Qwen3-Embedding-0.6B is a ~600M-param
decoder, much heavier than a MiniLM-style encoder, so raise this only if
your machine has memory to spare), `--learning-rate`, `--eval-ratio`,
`--seed`, `--max-seq-length` (default 512; longer inputs are truncated —
Qwen3-Embedding's own native context is far longer, 32768, but attention
cost is quadratic in sequence length, so raising this trades speed for less
truncation on long documents), and `--demo-text "..."` to run the
before/after comparison right after training. For the unsupervised
text-pool schema only: `--max-rows` (default 100,000, how many rows to
sample), `--no-english-filter` (disable the English-only filter), and
`--oversample-factor` (default 3, how many extra rows to sample before
language-filtering down to `--max-rows`).

The fine-tuned model is saved to `<output-dir>/final`.

### Notes on decoder-based embedding models (e.g. Qwen3-Embedding)

`--base-model` accepts any sentence-transformers-compatible model.
`load_model()` always forces `torch_dtype=float32` regardless of the
checkpoint's native dtype — Qwen3-Embedding ships as `bfloat16`, and
full-parameter fine-tuning directly in bf16 produced `NaN` eval loss on the
MPS backend in testing (MPS's bf16 op coverage/precision is less mature
than CUDA's). If you see `NaN` in TensorBoard after swapping to a different
base model, this is already handled here, but worth knowing about.

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
