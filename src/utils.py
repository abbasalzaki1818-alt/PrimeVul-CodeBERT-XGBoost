"""
utils.py
--------
Shared utilities used across every stage of the PrimeVul CodeBERT-XGBoost
pipeline: logging, reproducibility (seeding), path management, and generic
I/O helpers (JSONL / JSON / NumPy).

Design rule (per project constitution):
    - Nothing in this file assumes dataset field names or label semantics.
    - All paths are resolved relative to the project root so the code is
      portable across machines (PyCharm, terminal, CI, etc.).
"""

from __future__ import annotations

import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"

for _p in (DATASET_DIR, EMBEDDINGS_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR):
    _p.mkdir(parents=True, exist_ok=True)

RAW_SPLIT_FILES = {
    "train": DATASET_DIR / "primevul_train.jsonl",
    "valid": DATASET_DIR / "primevul_valid.jsonl",
    "test": DATASET_DIR / "primevul_test.jsonl",
}

PAIRED_SPLIT_FILES = {
    "train": DATASET_DIR / "primevul_train_paired.jsonl",
    "valid": DATASET_DIR / "primevul_valid_paired.jsonl",
    "test": DATASET_DIR / "primevul_test_paired.jsonl",
}

RANDOM_SEED_DEFAULT = 42


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Return a configured logger that writes to stdout and, optionally, a file.

    Using a shared logger factory (instead of `print`) keeps every module's
    output uniform and makes it trivial to redirect logs into the outputs
    directory for reproducibility records.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers on repeated calls
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_file:
        log_path = OUTPUTS_DIR / "metrics" / log_file
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

def set_global_seed(seed: int = RANDOM_SEED_DEFAULT) -> None:
    """Seed every RNG we can reach so experiments are reproducible.

    Torch / XGBoost are imported lazily (inside try/except) because not every
    stage of the pipeline needs them, and we don't want data_inspection.py
    (Stage 1) to require heavy ML dependencies just to run.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# JSONL / JSON I/O
# --------------------------------------------------------------------------- #

def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream a .jsonl file record by record.

    Streaming (rather than loading the whole file into memory) matters here
    because PrimeVul's benign split alone has ~229k records; we never want a
    single accidental `list(iter_jsonl(...))` call to blow up RAM before the
    user opts into it explicitly.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Expected dataset file not found: {path}\n"
            f"Place the official PrimeVul .jsonl files inside: {DATASET_DIR}"
        )
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON on line {line_number} of {path.name}: {exc}"
                ) from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Materialize a full .jsonl file into a list. Use only when the file is
    known to be small (e.g. after filtering), never for the raw split files
    without good reason."""
    return list(iter_jsonl(path))


def save_json(obj: Any, path: Path, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=_json_default)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    """Fallback serializer so numpy scalar types don't crash json.dump."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


# --------------------------------------------------------------------------- #
# NumPy embedding cache I/O
# --------------------------------------------------------------------------- #

def save_embeddings(split_name: str, embeddings: np.ndarray, labels: np.ndarray) -> None:
    np.save(EMBEDDINGS_DIR / f"{split_name}_embeddings.npy", embeddings)
    np.save(EMBEDDINGS_DIR / f"{split_name}_labels.npy", labels)


def load_embeddings(split_name: str) -> tuple[np.ndarray, np.ndarray]:
    emb_path = EMBEDDINGS_DIR / f"{split_name}_embeddings.npy"
    lbl_path = EMBEDDINGS_DIR / f"{split_name}_labels.npy"
    if not emb_path.exists() or not lbl_path.exists():
        raise FileNotFoundError(
            f"Cached embeddings for split '{split_name}' not found in {EMBEDDINGS_DIR}.\n"
            f"Run Stage 3 (embedding_extractor.py) before this stage."
        )
    return np.load(emb_path), np.load(lbl_path)


# --------------------------------------------------------------------------- #
# Small shared config object
# --------------------------------------------------------------------------- #

@dataclass
class DatasetSchema:
    """Resolved dataset schema, produced by Stage 1 (data_inspection) and
    consumed by every later stage. Kept as an explicit, saved artifact
    (outputs/metrics/dataset_schema.json) instead of hard-coded constants,
    so no later stage silently assumes field names.
    """
    code_field: str
    label_field: str
    label_mapping: dict[str, int]  # e.g. {"benign": 0, "vulnerable": 1}
    max_sequence_length: int = 512

    @staticmethod
    def load() -> "DatasetSchema":
        data = load_json(METRICS_DIR / "dataset_schema.json")
        return DatasetSchema(**data)

    def save(self) -> None:
        save_json(self.__dict__, METRICS_DIR / "dataset_schema.json")


def chunked(iterable: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """Yield fixed-size chunks from an iterable (used for CodeBERT batching)."""
    chunk: list[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
