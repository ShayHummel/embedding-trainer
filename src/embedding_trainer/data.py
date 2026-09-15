"""CSV loading and dataset preparation.

Supports three input schemas, auto-detected from the CSV header:

1. Pair format: two text columns (``sentence1``/``sentence2`` or
   ``text1``/``text2``) plus a numeric ``score`` (or ``similarity``) column,
   or a binary ``label`` column. Used directly for pair-based training.
2. Text + label format: a single ``text`` column plus a categorical
   ``label`` column. Positive (same-label) and negative (different-label)
   pairs are sampled to build a pair dataset with the same shape as (1).
3. Unsupervised text pool: a single free-text column (``text``, ``content``,
   or ``sentence``) with no label/score, e.g. a large unlabeled scrape. A
   random (optionally English-only) sample is drawn via reservoir sampling
   and trained with a SimCSE-style in-batch-negatives objective: each
   sentence is duplicated as its own "positive" and relies on dropout noise
   across the two forward passes to differ. A held-out slice is turned into
   (sentence1, sentence2, score) pairs (self-pairs vs. random pairs) purely
   for evaluation, so it plugs into the same evaluator as (1) and (2).
"""

import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import Dataset
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # deterministic language detection

PAIR_COLUMN_ALIASES = [("sentence1", "sentence2"), ("text1", "text2")]
SCORE_COLUMN_ALIASES = ["score", "similarity"]
LABEL_COLUMN = "label"
TEXT_COLUMN = "text"
UNSUPERVISED_TEXT_ALIASES = ["text", "content", "sentence"]


def _find_column(columns: list[str], names: list[str]) -> str | None:
    lower_to_actual = {c.lower(): c for c in columns}
    for name in names:
        if name in lower_to_actual:
            return lower_to_actual[name]
    return None


def _peek_columns(csv_path: Path) -> list[str]:
    return pd.read_csv(csv_path, nrows=0).columns.tolist()


def _has_pair_or_text_label_schema(columns: list[str]) -> bool:
    for col_a, col_b in PAIR_COLUMN_ALIASES:
        if _find_column(columns, [col_a]) and _find_column(columns, [col_b]):
            return True
    return bool(_find_column(columns, [TEXT_COLUMN]) and _find_column(columns, [LABEL_COLUMN]))


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


def _build_eval_pairs_from_texts(texts: list[str], seed: int = 42) -> pd.DataFrame:
    """Turn a flat list of texts into (sentence1, sentence2, score) eval pairs:
    each text paired with itself (score=1.0) and with one random other text
    (score=0.0)."""
    rng = random.Random(seed)
    n = len(texts)
    sentence1, sentence2, score = [], [], []
    for i, text in enumerate(texts):
        sentence1.append(text)
        sentence2.append(text)
        score.append(1.0)

        others = [j for j in range(n) if j != i]
        if others:
            j = rng.choice(others)
            sentence1.append(text)
            sentence2.append(texts[j])
            score.append(0.0)

    return pd.DataFrame({"sentence1": sentence1, "sentence2": sentence2, "score": score})


def load_pair_dataframe(csv_path: Path, seed: int = 42) -> tuple[pd.DataFrame, bool]:
    """Load a CSV and normalize it into a (sentence1, sentence2, score) frame.

    Returns the normalized dataframe and a flag indicating whether the score
    column is binary (0/1, from labels or a binary score) versus continuous.
    """
    df = pd.read_csv(csv_path)
    columns = df.columns.tolist()

    for col_a, col_b in PAIR_COLUMN_ALIASES:
        a = _find_column(columns, [col_a])
        b = _find_column(columns, [col_b])
        if a and b:
            score_col = _find_column(columns, SCORE_COLUMN_ALIASES) or _find_column(columns, [LABEL_COLUMN])
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

    text_col = _find_column(columns, [TEXT_COLUMN])
    label_col = _find_column(columns, [LABEL_COLUMN])
    if text_col and label_col:
        out = _build_pairs_from_text_label(df, text_col, label_col, seed=seed)
        return out, True

    raise ValueError(
        f"Could not detect a supported schema in {csv_path}. Expected either:\n"
        "  - pair columns: sentence1/sentence2 (or text1/text2) + score/similarity/label\n"
        "  - text + label columns: text, label\n"
        f"Found columns: {columns}"
    )


def _is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def _reservoir_sample_texts(
    csv_path: Path,
    text_col: str,
    target_size: int,
    seed: int = 42,
    chunksize: int = 50_000,
    min_chars: int = 20,
) -> list[str]:
    """Uniformly sample up to `target_size` non-trivial texts from a
    (potentially huge) CSV in a single streaming pass, via Algorithm R."""
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0
    for chunk in pd.read_csv(csv_path, usecols=[text_col], chunksize=chunksize, dtype=str):
        for text in chunk[text_col]:
            if not isinstance(text, str):
                continue
            text = text.strip()
            if len(text) < min_chars:
                continue
            if len(reservoir) < target_size:
                reservoir.append(text)
            else:
                j = rng.randint(0, seen)
                if j < target_size:
                    reservoir[j] = text
            seen += 1
    return reservoir


def load_unsupervised_train_eval_datasets(
    csv_path: Path,
    text_col: str,
    max_rows: int = 100_000,
    eval_ratio: float = 0.1,
    seed: int = 42,
    english_only: bool = True,
    oversample_factor: int = 3,
    chunksize: int = 50_000,
) -> tuple[Dataset, Dataset]:
    """Sample a labeled-pair-free text pool and prepare it for SimCSE-style
    unsupervised training: (train_dataset[sentence1, sentence2], eval_dataset
    [sentence1, sentence2, score])."""
    target = max_rows * oversample_factor if english_only else max_rows
    texts = _reservoir_sample_texts(csv_path, text_col, target_size=target, seed=seed, chunksize=chunksize)
    texts = list(dict.fromkeys(texts))  # de-duplicate, preserve order

    if english_only:
        texts = [t for t in texts if _is_english(t)]

    rng = random.Random(seed)
    rng.shuffle(texts)
    texts = texts[:max_rows]

    if len(texts) < 2:
        raise ValueError(
            f"Only found {len(texts)} usable row(s) in column '{text_col}' of {csv_path} "
            "after sampling/filtering; need at least 2 to build train/eval sets. Try a "
            "larger --max-rows/--oversample-factor, or disable the English-only filter."
        )

    split_idx = max(1, int(len(texts) * (1 - eval_ratio)))
    train_texts, eval_texts = texts[:split_idx], texts[split_idx:]
    if not eval_texts:
        eval_texts = train_texts[-1:]
        train_texts = train_texts[:-1]

    train_dataset = Dataset.from_dict({"sentence1": train_texts, "sentence2": train_texts})
    eval_dataset = Dataset.from_pandas(_build_eval_pairs_from_texts(eval_texts, seed=seed), preserve_index=False)
    return train_dataset, eval_dataset


def load_dataset_for_training(
    csv_path: Path,
    eval_ratio: float = 0.1,
    seed: int = 42,
    max_rows: int = 100_000,
    english_only: bool = True,
    oversample_factor: int = 3,
    chunksize: int = 50_000,
) -> tuple[Dataset, Dataset, str]:
    """Auto-detect the CSV schema and return (train_dataset, eval_dataset,
    training_format), where training_format is one of "cosine_similarity",
    "contrastive", or "unsupervised"."""
    columns = _peek_columns(csv_path)

    if _has_pair_or_text_label_schema(columns):
        df, is_binary = load_pair_dataframe(csv_path, seed=seed)
        dataset = Dataset.from_pandas(df, preserve_index=False)
        split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
        training_format = "contrastive" if is_binary else "cosine_similarity"
        return split["train"], split["test"], training_format

    text_only_col = _find_column(columns, UNSUPERVISED_TEXT_ALIASES)
    if text_only_col:
        train_dataset, eval_dataset = load_unsupervised_train_eval_datasets(
            csv_path,
            text_col=text_only_col,
            max_rows=max_rows,
            eval_ratio=eval_ratio,
            seed=seed,
            english_only=english_only,
            oversample_factor=oversample_factor,
            chunksize=chunksize,
        )
        return train_dataset, eval_dataset, "unsupervised"

    raise ValueError(
        f"Could not detect a supported schema in {csv_path}. Expected one of:\n"
        "  - pair columns: sentence1/sentence2 (or text1/text2) + score/similarity/label\n"
        "  - text + label columns: text, label\n"
        "  - a free-text column with no label: text, content, or sentence\n"
        f"Found columns: {columns}"
    )
