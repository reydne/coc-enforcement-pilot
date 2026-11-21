"""Compare the LLM outputs with the annotations by Ehsani et al. (MSR 2024).

Usage: python analyze_pilot.py <responses folder> <pilot_sample.csv> <annotated_issue_level.csv>
The responses folder must contain one .txt file per batch, with one JSON object per line.
"""
import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (classification_report, cohen_kappa_score,
                             confusion_matrix, precision_recall_fscore_support)

resp_dir, sample_csv, issue_csv = sys.argv[1], sys.argv[2], sys.argv[3]
TBDF = ["None", "Bitter frustration", "Impatience", "Mocking", "Irony", "Vulgarity",
        "Threat", "Entitlement", "Insulting", "Identity attacks/Name-Calling"]

# 1. Read the responses (tolerates extra text or code fences around the JSON)
rows, invalid = [], []
for f in sorted(Path(resp_dir).glob("*.txt")):
    for line in f.read_text(encoding="utf-8").splitlines():
        m = re.search(r"\{.*\}", line)
        if not m:
            continue
        try:
            rows.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            invalid.append((f.name, line[:80]))
llm = pd.DataFrame(rows).drop_duplicates("item", keep="last")
ref = pd.read_csv(sample_csv)
ref["tbdf_ref"] = ref.tbdf_ref.fillna("None")  # pandas reads the string "None" as missing
if llm.empty:
    sys.exit("No JSON responses found in the given folder.")
df = ref.merge(llm, on="item", how="left")
missing = df[df.tbdf.isna()].item.tolist()
df = df[df.tbdf.notna()].copy()
df["tbdf"] = df.tbdf.where(df.tbdf.isin(TBDF), "Invalid")
df["violation_llm"] = df.violation.astype(str).str.lower().eq("true").astype(int)

print(f"Items evaluated: {len(df)} of {len(ref)}. Missing: {missing}. Invalid lines: {len(invalid)}")

# 2. Task A: incivility feature (TBDF), single label
print("\n=== Task A: incivility feature (TBDF) ===")
print(f"Accuracy: {(df.tbdf == df.tbdf_ref).mean():.3f}")
print(f"Cohen's kappa (LLM vs. dataset): {cohen_kappa_score(df.tbdf_ref, df.tbdf):.3f}")
print(classification_report(df.tbdf_ref, df.tbdf, labels=TBDF, zero_division=0, digits=3))
cm = pd.DataFrame(confusion_matrix(df.tbdf_ref, df.tbdf, labels=TBDF), index=TBDF, columns=TBDF)
cm.to_csv("confusion_matrix_tbdf.csv")

# Binary: uncivil vs. civil
unc_ref = (df.tbdf_ref != "None").astype(int)
unc_llm = (df.tbdf != "None").astype(int)
p, r, f1, _ = precision_recall_fscore_support(unc_ref, unc_llm, average="binary", zero_division=0)
print(f"Uncivil vs. civil -> P={p:.3f} R={r:.3f} F1={f1:.3f} kappa={cohen_kappa_score(unc_ref, unc_llm):.3f}")

# 3. Task B: CoC violation against the a priori mapping TBDF -> violation
print("\n=== Task B: CoC violation (reference = a priori mapping) ===")
p, r, f1, _ = precision_recall_fscore_support(df.violation_ref, df.violation_llm, average="binary", zero_division=0)
print(f"P={p:.3f} R={r:.3f} F1={f1:.3f} kappa={cohen_kappa_score(df.violation_ref, df.violation_llm):.3f}")
print("Share flagged as violation by the LLM, per reference TBDF:")
print(df.groupby("tbdf_ref").violation_llm.agg(["mean", "count"]).round(2).sort_values("mean"))

# 4. Model confidence vs. correctness
if "confidence" in df:
    df["tbdf_correct"] = (df.tbdf == df.tbdf_ref).astype(int)
    print("\nTBDF accuracy per confidence level:")
    print(df.groupby("confidence").tbdf_correct.agg(["mean", "count"]).round(2))

# 5. RQ1 (no LLM): consequences annotated in the 404 locked threads
iss = pd.read_csv(issue_csv)
iss.columns = [c.strip() for c in iss.columns]
cons = iss.consequences.fillna("[]").apply(lambda v: ast.literal_eval(v) if v.startswith("[") else [])
n = len(iss)
print("\n=== RQ1: consequences in the locked threads (whole dataset) ===")
for k in ["Invoke Code of Conduct", "Trying to stop the incivility", "Discontinued further discussion", "Escalating further"]:
    k_n = cons.apply(lambda lst: k in lst).sum()
    print(f"{k}: {k_n} of {n} threads ({k_n / n:.1%})")

# 6. Disagreements for error analysis
cols = ["item", "comment_id", "repo", "tbdf_ref", "tbdf", "violation_ref", "violation_llm", "confidence", "rationale", "comment_body"]
df[df.tbdf != df.tbdf_ref][cols].to_csv("disagreements.csv", index=False)
print("\nFiles written: confusion_matrix_tbdf.csv, disagreements.csv")
