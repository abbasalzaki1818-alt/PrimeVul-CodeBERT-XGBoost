import csv
import json
from pathlib import Path

import numpy as np


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\Msi-x99\PycharmProjects\PrimeVul_CodeBERT_XGBoost"
)

TEST_PATH = (
    PROJECT_DIR
    / "dataset"
    / "primevul_test.jsonl"
)

PAIRED_TEST_PATH = (
    PROJECT_DIR
    / "dataset"
    / "primevul_test_paired.jsonl"
)

PREDICTIONS_PATH = (
    PROJECT_DIR
    / "outputs"
    / "metrics"
    / "predictions"
    / "Proposed_CodeBERT_XGBoost_test_predictions.npz"
)

OUTPUT_JSON = (
    PROJECT_DIR
    / "outputs"
    / "metrics"
    / "stage8_paired_evaluation.json"
)

OUTPUT_CSV = (
    PROJECT_DIR
    / "outputs"
    / "metrics"
    / "stage8_paired_details.csv"
)


# ============================================================
# Helper: load JSONL
# ============================================================

def load_jsonl(path: Path):
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


# ============================================================
# Load original TEST and paired TEST
# ============================================================

test_records = load_jsonl(TEST_PATH)
paired_records = load_jsonl(PAIRED_TEST_PATH)

print("=" * 90)
print("PRIMEVUL FINAL PAIRED EVALUATION")
print("=" * 90)

print("Original test records:", len(test_records))
print("Paired test records  :", len(paired_records))


# ============================================================
# Load FINAL proposed-model TEST predictions
# ============================================================

if not PREDICTIONS_PATH.exists():
    raise FileNotFoundError(
        f"Prediction file not found:\n{PREDICTIONS_PATH}"
    )

prediction_data = np.load(PREDICTIONS_PATH)

y_true = prediction_data["y_true"]
y_proba = prediction_data["y_proba"]

threshold = float(
    prediction_data["threshold"][0]
)

y_pred = (
    y_proba >= threshold
).astype(int)

print(f"Final threshold      : {threshold:.6f}")


# ============================================================
# Sanity check 1:
# prediction array must match original TEST size
# ============================================================

if len(test_records) != len(y_true):
    raise ValueError(
        "Original TEST size does not match saved "
        "prediction array size."
    )


# ============================================================
# Sanity check 2:
# saved labels must match original TEST labels exactly
# ============================================================

raw_test_labels = np.array(
    [
        int(record["target"])
        for record in test_records
    ],
    dtype=int,
)

if not np.array_equal(
    raw_test_labels,
    y_true.astype(int),
):
    raise ValueError(
        "Saved prediction labels do not match "
        "the original TEST ordering."
    )

print("TEST ordering check   : PASSED")


# ============================================================
# Build mapping:
# PrimeVul idx -> position inside TEST prediction array
# ============================================================

idx_to_position = {}

for position, record in enumerate(test_records):

    idx = record.get("idx")

    if idx in idx_to_position:
        raise ValueError(
            f"Duplicate idx detected in TEST: {idx}"
        )

    idx_to_position[idx] = position


# ============================================================
# Check paired file size
# ============================================================

if len(paired_records) % 2 != 0:
    raise ValueError(
        "Paired test file contains an odd number "
        "of records."
    )

n_pairs = len(paired_records) // 2


# ============================================================
# PrimeVul pair-wise outcome counters
# ============================================================

counts = {
    "P-C": 0,
    "P-V": 0,
    "P-B": 0,
    "P-R": 0,
}

pair_details = []


# ============================================================
# Evaluate every pair
#
# Every pair is:
#   first  = Vulnerable (target=1)
#   second = Benign/Patch (target=0)
# ============================================================

for pair_number in range(n_pairs):

    i = pair_number * 2

    vulnerable = paired_records[i]
    benign = paired_records[i + 1]

    # --------------------------------------------------------
    # Validate pair structure again
    # --------------------------------------------------------

    if int(vulnerable["target"]) != 1:
        raise ValueError(
            f"Pair {pair_number}: first element "
            "is not vulnerable."
        )

    if int(benign["target"]) != 0:
        raise ValueError(
            f"Pair {pair_number}: second element "
            "is not benign."
        )

    if (
        vulnerable.get("big_vul_idx")
        != benign.get("big_vul_idx")
    ):
        raise ValueError(
            f"Pair {pair_number}: "
            "big_vul_idx mismatch."
        )

    if (
        vulnerable.get("project")
        != benign.get("project")
    ):
        raise ValueError(
            f"Pair {pair_number}: project mismatch."
        )

    if (
        vulnerable.get("commit_id")
        != benign.get("commit_id")
    ):
        raise ValueError(
            f"Pair {pair_number}: commit mismatch."
        )

    vulnerable_idx = vulnerable["idx"]
    benign_idx = benign["idx"]

    if vulnerable_idx not in idx_to_position:
        raise ValueError(
            f"Vulnerable idx not found: "
            f"{vulnerable_idx}"
        )

    if benign_idx not in idx_to_position:
        raise ValueError(
            f"Benign idx not found: {benign_idx}"
        )

    # --------------------------------------------------------
    # Locate predictions from ORIGINAL TEST array
    # --------------------------------------------------------

    vulnerable_position = (
        idx_to_position[vulnerable_idx]
    )

    benign_position = (
        idx_to_position[benign_idx]
    )

    vulnerable_probability = float(
        y_proba[vulnerable_position]
    )

    benign_probability = float(
        y_proba[benign_position]
    )

    vulnerable_prediction = int(
        y_pred[vulnerable_position]
    )

    benign_prediction = int(
        y_pred[benign_position]
    )

    # --------------------------------------------------------
    # PrimeVul pair-wise classification
    # --------------------------------------------------------

    if (
        vulnerable_prediction == 1
        and benign_prediction == 0
    ):
        outcome = "P-C"

    elif (
        vulnerable_prediction == 1
        and benign_prediction == 1
    ):
        outcome = "P-V"

    elif (
        vulnerable_prediction == 0
        and benign_prediction == 0
    ):
        outcome = "P-B"

    elif (
        vulnerable_prediction == 0
        and benign_prediction == 1
    ):
        outcome = "P-R"

    else:
        raise RuntimeError(
            "Unexpected paired prediction state."
        )

    counts[outcome] += 1

    pair_details.append(
        {
            "pair_number": pair_number + 1,
            "big_vul_idx": vulnerable.get(
                "big_vul_idx"
            ),
            "project": vulnerable.get(
                "project"
            ),
            "commit_id": vulnerable.get(
                "commit_id"
            ),
            "vulnerable_idx": vulnerable_idx,
            "benign_idx": benign_idx,
            "vulnerable_probability":
                vulnerable_probability,
            "benign_probability":
                benign_probability,
            "vulnerable_prediction":
                vulnerable_prediction,
            "benign_prediction":
                benign_prediction,
            "pair_outcome": outcome,
        }
    )


# ============================================================
# Convert counts to percentages
# ============================================================

percentages = {
    key: (
        value / n_pairs * 100.0
    )
    for key, value in counts.items()
}


# ============================================================
# Final sanity check
# ============================================================

total_classified_pairs = sum(
    counts.values()
)

if total_classified_pairs != n_pairs:
    raise RuntimeError(
        "Pair outcome counts do not sum "
        "to total number of pairs."
    )


# ============================================================
# Save final JSON result
# ============================================================

results = {
    "model_name":
        "Proposed_CodeBERT_XGBoost",

    "threshold_used":
        threshold,

    "paired_test_records":
        len(paired_records),

    "total_pairs":
        n_pairs,

    "counts":
        counts,

    "percentages":
        percentages,

    "interpretation": {
        "P-C":
            "Both vulnerable and benign elements "
            "classified correctly.",

        "P-V":
            "Both elements predicted vulnerable.",

        "P-B":
            "Both elements predicted benign.",

        "P-R":
            "Predictions reversed relative to "
            "the ground truth.",
    },
}

OUTPUT_JSON.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT_JSON.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# Save pair-level CSV for reproducibility
# ============================================================

with OUTPUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=pair_details[0].keys(),
    )

    writer.writeheader()
    writer.writerows(pair_details)


# ============================================================
# Print final results
# ============================================================

print("\n" + "=" * 90)
print("PAIR-WISE RESULTS")
print("=" * 90)

print(f"Total pairs : {n_pairs}")

for key in ["P-C", "P-V", "P-B", "P-R"]:

    print(
        f"{key}: "
        f"{counts[key]:4d} pairs "
        f"({percentages[key]:.2f}%)"
    )

print("\nCheck sum:", total_classified_pairs)

print("\nJSON saved to:")
print(OUTPUT_JSON)

print("\nPair-level CSV saved to:")
print(OUTPUT_CSV)

print("\n" + "=" * 90)
print("PAIRED EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 90)