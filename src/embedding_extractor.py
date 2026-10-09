"""
embedding_extractor.py
-----------------------
STAGE 3 — Tokenization + CodeBERT Feature Extraction.

CodeBERT is used as a FROZEN feature extractor (no fine-tuning here — that
is reserved for an optional later experiment if GPU resources allow, per
the project notes). For each function:

    Source Code -> CodeBERT Tokenizer -> Tokens -> CodeBERT -> Embedding

We process data in batches, under `torch.no_grad()` / `model.eval()`, and
cache the resulting embeddings to outputs/embeddings/*.npy so CodeBERT never
needs to run again while XGBoost is being tuned.

Pooling strategy: mean-pooling over the last hidden state (excluding padding
tokens), which is a stronger sentence/function-level representation than the
raw [CLS] token for downstream classifiers built on top of frozen embeddings.

Run:
    python src/embedding_extractor.py
    python src/embedding_extractor.py --batch-size 16 --max-length 512
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from utils import (
    OUTPUTS_DIR,
    METRICS_DIR,
    DatasetSchema,
    get_logger,
    save_embeddings,
    save_json,
    set_global_seed,
)

logger = get_logger("embedding_extractor", log_file="stage3_embedding_extraction.log")

CLEANED_DIR = OUTPUTS_DIR / "cleaned"
DEFAULT_MODEL_NAME = "microsoft/codebert-base"


def check_device() -> str:
    """Inspect the machine and report what CodeBERT will run on.
    Per project notes: PrimeVul's original experiments used A100 GPUs, so we
    warn loudly if we're falling back to CPU on a large split."""
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info("GPU detected: %s (%.1f GB VRAM)", name, vram_gb)
        return "cuda"

    logger.warning(
        "No CUDA GPU detected. Running CodeBERT on CPU. This will be significantly "
        "slower on large splits (PrimeVul benign class is ~229k functions). "
        "Consider reducing --max-length or running on a machine with a GPU."
    )
    return "cpu"


def _read_clean_jsonl(path: Path) -> tuple[list[str], list[int]]:
    codes, labels = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            codes.append(rec["code"])
            labels.append(int(rec["label"]))
    return codes, labels


def extract_embeddings_for_split(
    split_name: str,
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_length: int,
) -> dict:
    import torch

    clean_path = CLEANED_DIR / f"{split_name}_clean.jsonl"
    if not clean_path.exists():
        raise FileNotFoundError(
            f"{clean_path} not found. Run Stage 2 (preprocessing.py) first."
        )

    codes, labels = _read_clean_jsonl(clean_path)
    n = len(codes)
    logger.info("[%s] extracting embeddings for %d functions...", split_name, n)

    all_embeddings = np.zeros((n, model.config.hidden_size), dtype=np.float32)
    truncated_count = 0
    start_time = time.time()

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_codes = codes[start:start + batch_size]

            encoded = tokenizer(
                batch_codes,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
                return_length=True,
            )

            # Count how many examples in this batch actually got truncated,
            # by re-tokenizing without truncation just for length comparison.
            raw_lengths = tokenizer(
                batch_codes, truncation=False, padding=False
            )["input_ids"]
            truncated_count += sum(1 for ids in raw_lengths if len(ids) > max_length)

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # (B, T, H)

            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = summed / counts  # mean pooling over real tokens only

            all_embeddings[start:start + len(batch_codes)] = (
                mean_pooled.cpu().numpy().astype(np.float32)
            )

            if (start // batch_size) % 20 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    "[%s] processed %d/%d (%.1fs elapsed)",
                    split_name, min(start + batch_size, n), n, elapsed,
                )

    labels_arr = np.array(labels, dtype=np.int64)
    save_embeddings(split_name, all_embeddings, labels_arr)

    stats = {
        "split_name": split_name,
        "num_functions": n,
        "embedding_dim": int(model.config.hidden_size),
        "truncated_functions": truncated_count,
        "truncated_pct": round(100.0 * truncated_count / n, 2) if n else 0.0,
        "elapsed_seconds": round(time.time() - start_time, 2),
        "label_distribution": {
            "0_benign": int((labels_arr == 0).sum()),
            "1_vulnerable": int((labels_arr == 1).sum()),
        },
    }
    logger.info("[%s] done. %.2f%% of functions were truncated at max_length=%d.",
                split_name, stats["truncated_pct"], max_length)
    return stats


def run_embedding_extraction(
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 16,
    max_length: int = 512,
) -> None:
    set_global_seed()
    from transformers import AutoModel, AutoTokenizer

    schema = DatasetSchema.load()  # ensures Stage 2 ran and schema is fixed
    logger.info("Loaded schema: %s", schema)

    device = check_device()
    logger.info("Loading tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    all_stats = {}
    for split_name in ("train", "valid", "test"):
        all_stats[split_name] = extract_embeddings_for_split(
            split_name, tokenizer, model, device, batch_size, max_length
        )

    save_json(all_stats, METRICS_DIR / "stage3_embedding_extraction_report.json")
    logger.info("Stage 3 complete. Embeddings cached in outputs/embeddings/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: CodeBERT embedding extraction")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    run_embedding_extraction(args.model_name, args.batch_size, args.max_length)
