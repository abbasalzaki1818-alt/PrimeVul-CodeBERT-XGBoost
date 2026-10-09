"""
preprocessing.py
-----------------
STAGE 2 — Preprocessing.

Consumes the Stage-1 inspection report (outputs/metrics/stage1_data_inspection_report.json)
and:
    1. Resolves the final DatasetSchema (code_field, label_field, label_mapping)
       — using the inferred fields as a *proposal* that a human can override
       via CLI flags, never as a silent hard-coded assumption.
    2. Performs MINIMAL cleaning only:
         - drop records with missing code or missing label
         - normalize label values to {0, 1} ints via the resolved mapping
       We deliberately do NOT strip whitespace/comments/rename variables:
       CodeBERT's representation can be sensitive to exactly that kind of
       surface information, and PrimeVul's own splits are already
       deduplicated/chronologically split, so aggressive cleaning would
       only risk destroying signal or reintroducing leakage.
    3. Writes cleaned, schema-normalized JSONL files to outputs/ for the
       embedding extraction stage to consume, plus a preprocessing report.

Run:
    python src/preprocessing.py
    python src/preprocessing.py --code-field func --label-field target --positive-label 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from utils import (
    RAW_SPLIT_FILES,
    METRICS_DIR,
    OUTPUTS_DIR,
    DatasetSchema,
    get_logger,
    iter_jsonl,
    load_json,
    save_json,
)

logger = get_logger("preprocessing", log_file="stage2_preprocessing.log")

CLEANED_DIR = OUTPUTS_DIR / "cleaned"
CLEANED_DIR.mkdir(parents=True, exist_ok=True)


def resolve_schema(
    code_field: str | None,
    label_field: str | None,
    positive_label: str | None,
) -> DatasetSchema:
    """Resolve the dataset schema from Stage 1's report, allowing explicit
    CLI overrides. Refuses to guess silently if inference failed and no
    override was given — this is a hard stop by design (Rule: never invent
    a result before running the code / never assume field semantics)."""
    report_path = METRICS_DIR / "stage1_data_inspection_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            "Stage 1 report not found. Run `python src/data_inspection.py` first."
        )
    report = load_json(report_path)
    train_report = report["splits"]["train"]

    resolved_code_field = code_field or train_report["inferred_code_field"]
    resolved_label_field = label_field or train_report["inferred_label_field"]

    if not resolved_code_field:
        raise ValueError(
            "Could not resolve the source-code field. Inspect "
            f"{report_path} manually and pass --code-field explicitly."
        )
    if not resolved_label_field:
        raise ValueError(
            "Could not resolve the label field. Inspect "
            f"{report_path} manually and pass --label-field explicitly."
        )

    # Determine label mapping to {0,1}. If the field already looks binary
    # numeric (0/1), keep it as-is; otherwise require --positive-label.
    candidate_values = train_report["label_field_candidates"].get(resolved_label_field)
    if candidate_values is None:
        # inferred_label_field may not be in label_field_candidates if it was
        # itself an override; fall back to trusting the user / raw values.
        candidate_values = []

    values_set = set(candidate_values)
    if values_set <= {"0", "1"} or values_set <= {"0.0", "1.0"}:
        label_mapping = {"0": 0, "1": 1}
    else:
        if positive_label is None:
            raise ValueError(
                f"Label field '{resolved_label_field}' has non-binary-numeric "
                f"values {sorted(values_set)}. Pass --positive-label to specify "
                f"which value means 'Vulnerable'."
            )
        label_mapping = {v: (1 if v == positive_label else 0) for v in values_set} or {
            positive_label: 1
        }

    schema = DatasetSchema(
        code_field=resolved_code_field,
        label_field=resolved_label_field,
        label_mapping=label_mapping,
        max_sequence_length=512,
    )
    schema.save()
    logger.info("Resolved schema: %s", schema)
    return schema


def _normalize_label(raw_value: Any, schema: DatasetSchema) -> int | None:
    key = str(raw_value)
    if key in schema.label_mapping:
        return schema.label_mapping[key]
    # Fall back: numeric 0/1 already
    if raw_value in (0, 1):
        return int(raw_value)
    return None


def clean_split(split_name: str, path: Path, schema: DatasetSchema) -> dict[str, Any]:
    out_path = CLEANED_DIR / f"{split_name}_clean.jsonl"
    kept, dropped_missing_code, dropped_missing_label, dropped_unmapped_label = 0, 0, 0, 0

    import json

    with out_path.open("w", encoding="utf-8") as out_f:
        for record in iter_jsonl(path):
            code = record.get(schema.code_field)
            label_raw = record.get(schema.label_field)

            if not code or not isinstance(code, str) or not code.strip():
                dropped_missing_code += 1
                continue
            if label_raw is None or label_raw == "":
                dropped_missing_label += 1
                continue

            label = _normalize_label(label_raw, schema)
            if label is None:
                dropped_unmapped_label += 1
                continue

            out_f.write(json.dumps({"code": code, "label": label}) + "\n")
            kept += 1

    stats = {
        "split_name": split_name,
        "input_path": str(path),
        "output_path": str(out_path),
        "kept_records": kept,
        "dropped_missing_code": dropped_missing_code,
        "dropped_missing_label": dropped_missing_label,
        "dropped_unmapped_label": dropped_unmapped_label,
    }
    logger.info(
        "[%s] kept=%d dropped(code=%d, label=%d, unmapped=%d) -> %s",
        split_name, kept, dropped_missing_code, dropped_missing_label,
        dropped_unmapped_label, out_path,
    )
    return stats


def run_preprocessing(
    code_field: str | None = None,
    label_field: str | None = None,
    positive_label: str | None = None,
) -> None:
    schema = resolve_schema(code_field, label_field, positive_label)

    all_stats = {}
    for split_name, path in RAW_SPLIT_FILES.items():
        all_stats[split_name] = clean_split(split_name, path, schema)

    save_json(all_stats, METRICS_DIR / "stage2_preprocessing_report.json")
    logger.info("Stage 2 complete. Report saved to outputs/metrics/stage2_preprocessing_report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Preprocessing / schema resolution")
    parser.add_argument("--code-field", default=None, help="Override the inferred code field name")
    parser.add_argument("--label-field", default=None, help="Override the inferred label field name")
    parser.add_argument(
        "--positive-label",
        default=None,
        help="Which raw label value means 'Vulnerable' (required if the label "
        "field is not already binary 0/1)",
    )
    args = parser.parse_args()
    run_preprocessing(args.code_field, args.label_field, args.positive_label)
