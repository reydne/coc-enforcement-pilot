"""Draw the stratified pilot sample from the dataset by Ehsani et al. (MSR 2024).

Usage: python prepare_sample.py <dataset/ folder of the incivility-dataset repository> <output folder>
"""
import sys
from pathlib import Path

import pandas as pd

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

c = pd.read_csv(src / "annotated_comment_level.csv")
# Merge different spellings of the same feature
c["tbdf"] = c.tbdf.replace({"Identity attack/Name-Calling": "Identity attacks/Name-Calling",
                            "Identity Attack/Name-Calling": "Identity attacks/Name-Calling"})
c["comment_body"] = c.comment_body.fillna("")
c = c.sort_values(["issue_id", "id"])
c["prev_body"] = c.groupby("issue_id").comment_body.shift(1)  # preceding comment in the same issue
t = pd.read_csv(src / "issue_threads.csv")[["issue_id", "issue_title", "url"]]
c = c.merge(t, on="issue_id", how="left")
c["repo"] = c.url.str.extract(r"repos/([^/]+/[^/]+)/issues")

# Few-shot examples: these comments and their issues are excluded from the sample
FEWSHOT = [1229192711, 1088264852, 378762344, 1082171169, 1160417956, 1102331681]
excluded_issues = set(c[c.comment_id.isin(FEWSHOT)].issue_id)
pool = c[~c.issue_id.isin(excluded_issues) & (c.comment_body.str.strip().str.len() > 0)]

# 100 uncivil comments stratified by TBDF + 100 civil comments, fixed seeds
ALLOCATION = {"Bitter frustration": 20, "Impatience": 14, "Mocking": 14, "Insulting": 14, "Vulgarity": 9,
              "Entitlement": 9, "Irony": 9, "Threat": 6, "Identity attacks/Name-Calling": 5}
parts = [pool[pool.tbdf == k].sample(n, random_state=42) for k, n in ALLOCATION.items()]
parts.append(pool[pool.tbdf.isna()].sample(100, random_state=42))
s = pd.concat(parts).sample(frac=1, random_state=7).reset_index(drop=True)
s["item"] = range(1, len(s) + 1)
s["tbdf_ref"] = s.tbdf.fillna("None")

# A priori mapping from TBDF to CoC violation (Contributor Covenant)
VIOLATION = {"Insulting", "Mocking", "Irony", "Identity attacks/Name-Calling", "Vulgarity"}
s["violation_ref"] = s.tbdf_ref.isin(VIOLATION).astype(int)

cols = ["item", "comment_id", "issue_id", "repo", "issue_title", "tbdf_ref", "violation_ref", "comment_body", "prev_body"]
s[cols].to_csv(out / "pilot_sample.csv", index=False)
c[c.comment_id.isin(FEWSHOT)][["comment_id", "issue_id", "tbdf", "comment_body"]].to_csv(
    out / "fewshot_examples.csv", index=False)
print(s.tbdf_ref.value_counts().to_string())
print("Violations in the reference:", s.violation_ref.sum())
