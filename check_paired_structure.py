import json
from pathlib import Path

file_path = Path(
    r"C:\Users\Msi-x99\PycharmProjects\PrimeVul_CodeBERT_XGBoost"
    r"\dataset\primevul_test_paired.jsonl"
)

records = []

with file_path.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        records.append(json.loads(line))

        if i >= 9:
            break

print("=" * 100)
print("FIRST 10 RECORDS FROM primevul_test_paired.jsonl")
print("=" * 100)

for i, record in enumerate(records):
    print(f"\nRecord {i}")
    print("target      :", record.get("target"))
    print("idx         :", record.get("idx"))
    print("big_vul_idx :", record.get("big_vul_idx"))
    print("project     :", record.get("project"))
    print("commit_id   :", record.get("commit_id"))
    print("hash        :", record.get("hash"))

print("\n" + "=" * 100)
print("CONSECUTIVE PAIR CHECK")
print("=" * 100)

for i in range(0, len(records) - 1, 2):

    a = records[i]
    b = records[i + 1]

    print(
        f"Records {i} & {i + 1}: "
        f"targets=({a.get('target')}, {b.get('target')}) | "
        f"same project={a.get('project') == b.get('project')} | "
        f"same commit={a.get('commit_id') == b.get('commit_id')} | "
        f"big_vul_idx=({a.get('big_vul_idx')}, {b.get('big_vul_idx')})"
    )