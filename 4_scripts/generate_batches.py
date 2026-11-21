"""Build the five batches of 40 items sent to the model from the sample and the prompt header.

Usage: python generate_batches.py <pilot_sample.csv> <prompt_header_v1.txt> <output folder>
"""
import sys
from pathlib import Path

import pandas as pd

sample, header, out = sys.argv[1], Path(sys.argv[2]).read_text(encoding="utf-8"), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
s = pd.read_csv(sample)


def truncate(x, n):
    x = "" if pd.isna(x) else str(x).replace('"""', "'''").strip()
    return x if len(x) <= n else x[:n] + " [...truncated]"


for b in range(5):
    blocks = [header]
    for _, r in s.iloc[b * 40:(b + 1) * 40].iterrows():
        prev = truncate(r["prev_body"], 600) or "(none - this is the first post of the issue)"
        blocks.append(f'### Item {r["item"]}\nRepository: {r["repo"]}\nIssue title: {truncate(r["issue_title"], 200)}\n'
                      f'Previous comment:\n"""\n{prev}\n"""\nTARGET COMMENT:\n"""\n{truncate(r["comment_body"], 1500)}\n"""\n')
    (out / f"batch_{b + 1:02d}.txt").write_text("\n".join(blocks), encoding="utf-8")
print("5 batches written to", out)
