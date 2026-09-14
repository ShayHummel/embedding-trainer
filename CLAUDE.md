# Project: Embedding Model Trainer

## Goal
Build a program to train, test, and validate a sentence embedding model
end-to-end, runnable locally on Apple Silicon (M5).

## Requirements
- Base model: open-weight sentence embedding model (default: 
  sentence-transformers/all-MiniLM-L6-v2). Must be swappable.
- Language: English only.
- Data: user-provided CSV in data/, with a text column (schema TBD 
  once CSV is shared — likely pairs or text+label for contrastive loss).
- Must include a dedicated function `show_vector_before_after()` that 
  encodes the same input with the base model and the fine-tuned model, 
  and prints/plots both vectors + similarity score.
- Device: torch.device("mps") with CPU fallback. Set 
  PYTORCH_ENABLE_MPS_FALLBACK=1.
- Training must use sentence-transformers' SentenceTransformerTrainer 
  with TrainingArguments(report_to="tensorboard"), logging to runs/.
- Include an evaluator (e.g. EmbeddingSimilarityEvaluator) so 
  eval loss/correlation curves show up in TensorBoard during training.
- Env managed with uv (not pip/conda). All deps in pyproject.toml.
- Not using any OpenAI API model — must be fully local/offline.

## Out of scope
- Multilingual support
- Cloud/GPU training
- API-based embedding models (OpenAI, Cohere, etc.)