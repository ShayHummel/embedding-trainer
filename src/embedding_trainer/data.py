"""CSV loading and dataset preparation.

Supports two input schemas, auto-detected from the CSV header:

1. Pair format: two text columns (``sentence1``/``sentence2`` or
   ``text1``/``text2``) plus a numeric ``score`` (or ``similarity``) column,
   or a binary ``label`` column. Used directly for pair-based training.
2. Text + label format: a single ``text`` column plus a categorical
   ``label`` column. Positive (same-label) and negative (different-label)
   pairs are sampled to build a pair dataset with the same shape as (1),
   so both formats feed the same training/evaluation path.
"""

import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import Dataset

PAIR_COLUMN_ALIASES = [("sentence1", "sentence2"), ("text1", "text2")]
SCORE_COLUMN_ALIASES = ["score", "similarity"]
LABEL_COLUMN = "label"
TEXT_COLUMN = "text"


def _find_columns(df: pd.DataFrame, names: list[str]) -> str | None:
    lower_to_actual = {c.lower(): c for c in df.columns}
    for name in names:
        if name in lower_to_actual:
            return lower_to_actual[name]
    return None


def _build_pairs_from_text_label(
    df: pd.DataFrame, text_col: str, label_col: str, seed: int = 42
) -> pd.DataFrame:
    """Sample one positive and one negative pair per row from labeled text."""
    rng = random.Random(seed)
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()

    by_label: dict = defaultdict(list)
    for idx, lbl in enumerate(labels):
        by_label[lbl].append(idx)

    sentence1, sentence2, score = [], [], []
    for idx, lbl in enumerate(labels):
        same_label_idxs = [j for j in by_label[lbl] if j != idx]
        if same_label_idxs:
            j = rng.choice(same_label_idxs)
            sentence1.append(texts[idx])
            sentence2.append(texts[j])
            score.append(1.0)

        other_labels = [l for l in by_label if l != lbl]
        if other_labels:
            other_lbl = rng.choice(other_labels)
            j = rng.choice(by_label[other_lbl])
            sentence1.append(texts[idx])
            sentence2.append(texts[j])
            score.append(0.0)

    return pd.DataFrame({"sentence1": sentence1, "sentence2": sentence2, "score": score})


def load_pair_dataframe(csv_path: Path, seed: int = 42) -> tuple[pd.DataFrame, bool]:
    """Load a CSV and normalize it into a (sentence1, sentence2, score) frame.

    Returns the normalized dataframe and a flag indicating whether the score
    column is binary (0/1, from labels or a binary score) versus continuous.
    """
    df = pd.read_csv(csv_path)

    for col_a, col_b in PAIR_COLUMN_ALIASES:
        a = _find_columns(df, [col_a])
        b = _find_columns(df, [col_b])
        if a and b:
            score_col = _find_columns(df, SCORE_COLUMN_ALIASES) or _find_columns(df, [LABEL_COLUMN])
            if score_col is None:
                raise ValueError(
                    f"Found pair columns '{a}'/'{b}' in {csv_path}, but no "
                    f"'score', 'similarity', or 'label' column to pair them with."
                )
            out = pd.DataFrame(
                {
                    "sentence1": df[a],
                    "sentence2": df[b],
                    "score": df[score_col].astype(float),
                }
            )
            is_binary = out["score"].dropna().isin([0.0, 1.0]).all()
            if not is_binary and out["score"].max() > 1.0:
                out["score"] = out["score"] / out["score"].max()
            return out, is_binary

    text_col = _find_columns(df, [TEXT_COLUMN])
    label_col = _find_columns(df, [LABEL_COLUMN])
    if text_col and label_col:
        out = _build_pairs_from_text_label(df, text_col, label_col, seed=seed)
        return out, True

    raise ValueError(
        f"Could not detect a supported schema in {csv_path}. Expected either:\n"
        "  - pair columns: sentence1/sentence2 (or text1/text2) + score/similarity/label\n"
        "  - text + label columns: text, label\n"
        f"Found columns: {list(df.columns)}"
    )


def load_train_eval_datasets(
    csv_path: Path, eval_ratio: float = 0.1, seed: int = 42
) -> tuple[Dataset, Dataset, bool]:
    """Load, split, and return (train_dataset, eval_dataset, is_binary_score)."""
    df, is_binary = load_pair_dataframe(csv_path, seed=seed)
    dataset = Dataset.from_pandas(df, preserve_index=False)
    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
    return split["train"], split["test"], is_binary
