"""
data_inspection.py
-------------------
STAGE 1 of the pipeline — and the ONLY stage that must run before anything
else touches the data.

Per project rules, we do NOT assume:
    - what the source-code field is called (e.g. "func", "code", "source")
    - what the label field is called (e.g. "target", "label", "vul")
    - what 0 / 1 mean
    - that the schema is even consistent across train/valid/test

Instead, this module:
    1. Streams each split file and discovers every field name that appears.
    2. Infers, from the data itself, which field most likely holds source
       code (long strings with code-like tokens) and which field most
       likely holds the binary label (few unique values, e.g. {0, 1}).
    3. Reports record counts, missing values, exact-duplicate source code,
       and code-length statistics.
    4. Writes a full JSON report + a printed console summary to
       outputs/metrics/, so every later stage builds on *observed* facts,
       not assumptions.

Run:
    python src/data_inspection.py
    python src/data_inspection.py --paired   # also inspect *_paired.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from utils import (
    RAW_SPLIT_FILES,
    PAIRED_SPLIT_FILES,
    METRICS_DIR,
    get_logger,
    iter_jsonl,
    save_json,
)

logger = get_logger("data_inspection", log_file="stage1_data_inspection.log")

# A field is a *candidate label field* if every observed value is one of a
# small set of scalar values (numbers or short strings/bools).
MAX_UNIQUE_VALUES_FOR_LABEL_CANDIDATE = 10

# A field is a *candidate code field* if its values are strings whose median
# length is well above this threshold and which contain code-like tokens.
MIN_MEDIAN_LEN_FOR_CODE_CANDIDATE = 40
CODE_TOKEN_PATTERN = re.compile(r"[{};()]|->|::|#include|void |int |return ")


def _looks_like_code(sample_values: list[str]) -> float:
    """Return a 0-1 'code likelihood' score for a batch of string values."""
    if not sample_values:
        return 0.0
    hits = sum(1 for v in sample_values if CODE_TOKEN_PATTERN.search(v or ""))
    return hits / len(sample_values)


def _build_sample_record(field_value_samples: dict[str, list[Any]]) -> dict[str, Any]:
    """Build one human-readable example record (first sampled value per field),
    truncating long strings so the JSON report stays readable."""
    sample: dict[str, Any] = {}
    for field, values in field_value_samples.items():
        if not values:
            continue
        value = values[0]
        if isinstance(value, str) and len(value) > 300:
            value = value[:300] + "..."
        sample[field] = value
    return sample


def inspect_split(split_name: str, path: Path, sample_cap: int = 5000) -> dict[str, Any]:
    """Inspect a single split file and return a structured report.

    sample_cap bounds how many records we use for *field-level* statistics
    (dtype histograms, code-likelihood, string lengths) to keep inspection
    fast even on the ~229k-record benign-heavy split, while record COUNTS
    and duplicate/label counts are always computed over the FULL file.
    """
    logger.info("Inspecting split '%s' -> %s", split_name, path)

    total_records = 0
    field_presence: Counter[str] = Counter()
    field_missing: Counter[str] = Counter()
    field_value_samples: dict[str, list[Any]] = {}
    field_all_scalar_values: dict[str, set[Any]] = {}
    field_is_scalar: dict[str, bool] = {}

    code_hash_counts: Counter[str] = Counter()
    code_len_chars: list[int] = []
    code_len_lines: list[int] = []

    first_record_keys: set[str] | None = None
    schema_drift_records: list[int] = []

    for idx, record in enumerate(iter_jsonl(path)):
        total_records += 1
        keys = set(record.keys())

        if first_record_keys is None:
            first_record_keys = keys
        elif keys != first_record_keys:
            schema_drift_records.append(idx)

        for key, value in record.items():
            field_presence[key] += 1
            if value is None or value == "":
                field_missing[key] += 1

            if idx < sample_cap:
                field_value_samples.setdefault(key, []).append(value)

            # Track scalar-ness / cardinality for label-field inference.
            is_scalar = isinstance(value, (int, float, bool, str)) and not (
                isinstance(value, str) and len(value) > 200
            )
            field_is_scalar.setdefault(key, True)
            field_is_scalar[key] = field_is_scalar[key] and is_scalar
            if is_scalar:
                bucket = field_all_scalar_values.setdefault(key, set())
                if len(bucket) <= MAX_UNIQUE_VALUES_FOR_LABEL_CANDIDATE * 3:
                    bucket.add(value)

        # Missing fields relative to the first record's schema.
        for missing_key in (first_record_keys or set()) - keys:
            field_missing[missing_key] += 1

    all_fields = sorted(field_presence.keys())

    # ---- Infer code field -------------------------------------------------
    code_field_scores: dict[str, float] = {}
    for field in all_fields:
        samples = [v for v in field_value_samples.get(field, []) if isinstance(v, str)]
        if not samples:
            continue
        lengths = [len(v) for v in samples]
        median_len = float(np.median(lengths)) if lengths else 0.0
        if median_len < MIN_MEDIAN_LEN_FOR_CODE_CANDIDATE:
            continue
        code_likelihood = _looks_like_code(samples[:200])
        code_field_scores[field] = code_likelihood * np.log1p(median_len)

    inferred_code_field = (
        max(code_field_scores, key=code_field_scores.get) if code_field_scores else None
    )

    # ---- Infer label field --------------------------------------------------
    label_field_candidates: dict[str, set[Any]] = {}
    for field, values in field_all_scalar_values.items():
        if field == inferred_code_field:
            continue
        if 1 < len(values) <= MAX_UNIQUE_VALUES_FOR_LABEL_CANDIDATE:
            label_field_candidates[field] = values

    # Prefer a binary {0,1}-like field named suggestively, else smallest cardinality.
    def _label_score(item: tuple[str, set[Any]]) -> tuple[int, int]:
        name, values = item
        name_hint = any(
            hint in name.lower() for hint in ("target", "label", "vul", "class", "y")
        )
        return (0 if name_hint else 1, len(values))

    inferred_label_field = None
    if label_field_candidates:
        inferred_label_field = sorted(
            label_field_candidates.items(), key=_label_score
        )[0][0]

    # ---- Full-file pass: code length + duplicate detection + label distribution ----
    label_distribution: Counter[Any] = Counter()
    for record in iter_jsonl(path):
        if inferred_code_field and inferred_code_field in record:
            code_val = record[inferred_code_field] or ""
            code_len_chars.append(len(code_val))
            code_len_lines.append(code_val.count("\n") + 1 if code_val else 0)
            code_hash = hashlib.sha256(code_val.encode("utf-8", errors="ignore")).hexdigest()
            code_hash_counts[code_hash] += 1
        if inferred_label_field and inferred_label_field in record:
            label_distribution[record[inferred_label_field]] += 1

    duplicate_code_records = sum(c for c in code_hash_counts.values() if c > 1)
    unique_duplicate_groups = sum(1 for c in code_hash_counts.values() if c > 1)

    def _pct_stats(values: list[int]) -> dict[str, float]:
        if not values:
            return {}
        arr = np.array(values)
        return {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
        }

    report = {
        "split_name": split_name,
        "file_path": str(path),
        "total_records": total_records,
        "fields_detected": all_fields,
        "field_presence_counts": dict(field_presence),
        "field_missing_counts": dict(field_missing),
        "schema_drift_record_indices_sample": schema_drift_records[:20],
        "schema_drift_record_count": len(schema_drift_records),
        "inferred_code_field": inferred_code_field,
        "inferred_label_field": inferred_label_field,
        "label_field_candidates": {
            k: sorted(map(str, v)) for k, v in label_field_candidates.items()
        },
        "label_distribution": {str(k): v for k, v in label_distribution.items()},
        "code_length_char_stats": _pct_stats(code_len_chars),
        "code_length_line_stats": _pct_stats(code_len_lines),
        "duplicate_code_records": duplicate_code_records,
        "unique_duplicate_groups": unique_duplicate_groups,
        "sample_record": _build_sample_record(field_value_samples),
    }

    logger.info(
        "[%s] records=%d fields=%s inferred_code_field=%s inferred_label_field=%s",
        split_name, total_records, all_fields, inferred_code_field, inferred_label_field,
    )
    if label_distribution:
        logger.info("[%s] label distribution: %s", split_name, dict(label_distribution))

    return report


def run_inspection(include_paired: bool = False) -> dict[str, Any]:
    full_report: dict[str, Any] = {"splits": {}, "paired_splits": {}}

    for split_name, path in RAW_SPLIT_FILES.items():
        full_report["splits"][split_name] = inspect_split(split_name, path)

    if include_paired:
        for split_name, path in PAIRED_SPLIT_FILES.items():
            if path.exists():
                full_report["paired_splits"][split_name] = inspect_split(
                    f"{split_name}_paired", path
                )
            else:
                logger.warning("Paired file not found, skipping: %s", path)

    # ---- Cross-split consistency checks ----
    code_fields = {
        s["inferred_code_field"] for s in full_report["splits"].values() if s["inferred_code_field"]
    }
    label_fields = {
        s["inferred_label_field"] for s in full_report["splits"].values() if s["inferred_label_field"]
    }
    full_report["consistency_check"] = {
        "consistent_code_field_across_splits": len(code_fields) <= 1,
        "code_fields_seen": sorted(code_fields),
        "consistent_label_field_across_splits": len(label_fields) <= 1,
        "label_fields_seen": sorted(label_fields),
    }

    out_path = METRICS_DIR / "stage1_data_inspection_report.json"
    save_json(full_report, out_path)
    logger.info("Full inspection report written to %s", out_path)

    _print_human_summary(full_report)
    return full_report


def _print_human_summary(report: dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("STAGE 1 — DATASET INSPECTION SUMMARY")
    print("=" * 78)
    for split_name, s in report["splits"].items():
        print(f"\n[{split_name.upper()}]  file: {s['file_path']}")
        print(f"  total records          : {s['total_records']:,}")
        print(f"  fields detected         : {s['fields_detected']}")
        print(f"  inferred code field     : {s['inferred_code_field']}")
        print(f"  inferred label field    : {s['inferred_label_field']}")
        print(f"  label distribution      : {s['label_distribution']}")
        print(f"  duplicate code records  : {s['duplicate_code_records']:,} "
              f"(in {s['unique_duplicate_groups']:,} groups)")
        if s["code_length_char_stats"]:
            cs = s["code_length_char_stats"]
            print(f"  code length (chars)     : median={cs['median']:.0f}, "
                  f"p90={cs['p90']:.0f}, p99={cs['p99']:.0f}, max={cs['max']:.0f}")
        if s["schema_drift_record_count"]:
            print(f"  !! schema drift detected in {s['schema_drift_record_count']} records")

    cc = report["consistency_check"]
    print("\n[CONSISTENCY CHECK ACROSS SPLITS]")
    print(f"  code field consistent   : {cc['consistent_code_field_across_splits']} "
          f"({cc['code_fields_seen']})")
    print(f"  label field consistent  : {cc['consistent_label_field_across_splits']} "
          f"({cc['label_fields_seen']})")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: PrimeVul dataset inspection")
    parser.add_argument(
        "--paired", action="store_true", help="Also inspect the *_paired.jsonl files"
    )
    args = parser.parse_args()
    run_inspection(include_paired=args.paired)
