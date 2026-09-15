"""Before/after embedding comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util


def show_vector_before_after(
    text: str,
    base_model: SentenceTransformer,
    fine_tuned_model: SentenceTransformer,
    plot_path: Path | None = Path("runs/vector_before_after.png"),
):
    """Encode `text` with both models and report/plot the two vectors and their similarity."""
    base_vec = base_model.encode(text)
    fine_tuned_vec = fine_tuned_model.encode(text)
    similarity = util.cos_sim(base_vec, fine_tuned_vec).item()

    print(f"Input text: {text!r}")
    print(f"Base model vector (dim={len(base_vec)}), first 10 values: {base_vec[:10]}")
    print(f"Fine-tuned vector (dim={len(fine_tuned_vec)}), first 10 values: {fine_tuned_vec[:10]}")
    print(f"Cosine similarity(base, fine-tuned): {similarity:.4f}")

    if plot_path is not None:
        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        axes[0].plot(base_vec)
        axes[0].set_title("Base model embedding")
        axes[1].plot(fine_tuned_vec)
        axes[1].set_title("Fine-tuned model embedding")
        axes[1].set_xlabel("Dimension")
        fig.suptitle(f"cosine similarity = {similarity:.4f}")
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Saved plot to {plot_path}")

    return base_vec, fine_tuned_vec, similarity
