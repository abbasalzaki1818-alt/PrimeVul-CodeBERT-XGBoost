import json
from pathlib import Path

PROJECT_DIR = Path(
    r"C:\Users\Msi-x99\PycharmProjects\PrimeVul_CodeBERT_XGBoost"
)

paired_path = (
    PROJECT_DIR
    / "dataset"
    / "primevul_test_paired.jsonl"
)

test_path = (
    PROJECT_DIR
    / "dataset"
    / "primevul_test.jsonl"
)


# ============================================================
# Load paired test data
# ============================================================

paired_records = []

with paired_path.open("r", encoding="utf-8") as f:
    for line in f:
        paired_records.append(json.loads(line))


print("=" * 90)
print("FULL PAIRED STRUCTURE CHECK")
print("=" * 90)

print("Total paired records:", len(paired_records))
print("Expected pairs:", len(paired_records) // 2)


# ============================================================
# Check every consecutive pair
# ============================================================

valid_pairs = 0
invalid_pairs = []

for i in range(0, len(paired_records), 2):

    if i + 1 >= len(paired_records):
        invalid_pairs.append(
            (i, "Missing second record")
        )
        continue

    vulnerable = paired_records[i]
    benign = paired_records[i + 1]

    checks = {
        "labels_correct":
            vulnerable.get("target") == 1
            and benign.get("target") == 0,

        "same_big_vul_idx":
            vulnerable.get("big_vul_idx")
            == benign.get("big_vul_idx"),

        "same_project":
            vulnerable.get("project")
            == benign.get("project"),

        "same_commit":
            vulnerable.get("commit_id")
            == benign.get("commit_id"),
    }

    if all(checks.values()):
        valid_pairs += 1
    else:
        invalid_pairs.append(
            {
                "records": (i, i + 1),
                "checks": checks,
                "first_idx": vulnerable.get("idx"),
                "second_idx": benign.get("idx"),
            }
        )


print("\nValid pairs:", valid_pairs)
print("Invalid pairs:", len(invalid_pairs))

if invalid_pairs:
    print("\nFirst invalid examples:")
    for item in invalid_pairs[:10]:
        print(item)


# ============================================================
# Check whether paired records occur in original test split
# ============================================================

test_idx_set = set()

with test_path.open("r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        test_idx_set.add(record.get("idx"))


paired_indices = [
    record.get("idx")
    for record in paired_records
]

found_indices = [
    idx
    for idx in paired_indices
    if idx in test_idx_set
]

missing_indices = [
    idx
    for idx in paired_indices
    if idx not in test_idx_set
]


print("\n" + "=" * 90)
print("PAIRED RECORD COVERAGE IN ORIGINAL TEST SPLIT")
print("=" * 90)

print("Paired records:", len(paired_indices))
print("Found in original test:", len(found_indices))
print("Missing from original test:", len(missing_indices))

coverage = (
    len(found_indices)
    / len(paired_indices)
    * 100
)

print(f"Coverage: {coverage:.2f}%")

if missing_indices:
    print("\nFirst missing idx values:")
    print(missing_indices[:20])


print("\n" + "=" * 90)

if (
    len(invalid_pairs) == 0
    and len(missing_indices) == 0
):
    print("RESULT: ALL CHECKS PASSED")
else:
    print("RESULT: CHECKS REQUIRE ATTENTION")

print("=" * 90)