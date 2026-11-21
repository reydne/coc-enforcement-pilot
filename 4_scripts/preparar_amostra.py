"""Gera a amostra estratificada do piloto a partir do dataset de Ehsani et al. (MSR 2024).

Uso: python preparar_amostra.py <pasta dataset/ do repositório incivility-dataset> <pasta de saída>
"""
import sys
from pathlib import Path
import pandas as pd

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

c = pd.read_csv(src / "annotated_comment_level.csv")
# Unifica grafias diferentes da mesma categoria
c["tbdf"] = c.tbdf.replace({"Identity attack/Name-Calling": "Identity attacks/Name-Calling",
                            "Identity Attack/Name-Calling": "Identity attacks/Name-Calling"})
c["comment_body"] = c.comment_body.fillna("")
c = c.sort_values(["issue_id", "id"])
c["prev_body"] = c.groupby("issue_id").comment_body.shift(1)  # comentário anterior na mesma issue
t = pd.read_csv(src / "issue_threads.csv")[["issue_id", "issue_title", "url"]]
c = c.merge(t, on="issue_id", how="left")
c["repo"] = c.url.str.extract(r"repos/([^/]+/[^/]+)/issues")

# Exemplos few-shot: os comentários e as issues deles ficam fora da amostra
FEWSHOT = [1229192711, 1088264852, 378762344, 1082171169, 1160417956, 1102331681]
excl_issues = set(c[c.comment_id.isin(FEWSHOT)].issue_id)
pool = c[~c.issue_id.isin(excl_issues) & (c.comment_body.str.strip().str.len() > 0)]

# 100 incivis estratificados por TBDF + 100 civis, semente fixa
ALLOC = {"Bitter frustration": 20, "Impatience": 14, "Mocking": 14, "Insulting": 14, "Vulgarity": 9,
         "Entitlement": 9, "Irony": 9, "Threat": 6, "Identity attacks/Name-Calling": 5}
parts = [pool[pool.tbdf == k].sample(n, random_state=42) for k, n in ALLOC.items()]
parts.append(pool[pool.tbdf.isna()].sample(100, random_state=42))
s = pd.concat(parts).sample(frac=1, random_state=7).reset_index(drop=True)
s["item"] = range(1, len(s) + 1)
s["tbdf_ref"] = s.tbdf.fillna("None")

# Mapeamento a priori TBDF -> violação de CoC (Contributor Covenant)
VIOLATION = {"Insulting", "Mocking", "Irony", "Identity attacks/Name-Calling", "Vulgarity"}
s["violation_ref"] = s.tbdf_ref.isin(VIOLATION).astype(int)

cols = ["item", "comment_id", "issue_id", "repo", "issue_title", "tbdf_ref", "violation_ref", "comment_body", "prev_body"]
s[cols].to_csv(out / "amostra_piloto.csv", index=False)
c[c.comment_id.isin(FEWSHOT)][["comment_id", "issue_id", "tbdf", "comment_body"]].to_csv(out / "fewshot_usados.csv", index=False)
print(s.tbdf_ref.value_counts().to_string())
print("Violações na referência:", s.violation_ref.sum())
