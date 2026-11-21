"""Gera os 5 lotes de 40 itens enviados ao modelo, a partir da amostra e do cabeçalho do prompt.

Uso: python gerar_lotes.py <amostra_piloto.csv> <cabecalho_prompt_v1.txt> <pasta de saída>
"""
import sys
from pathlib import Path
import pandas as pd

amostra, cabecalho, out = sys.argv[1], Path(sys.argv[2]).read_text(encoding="utf-8"), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
s = pd.read_csv(amostra)

def tr(x, n):
    x = "" if pd.isna(x) else str(x).replace('"""', "'''").strip()
    return x if len(x) <= n else x[:n] + " [...truncated]"

for b in range(5):
    blocos = [cabecalho]
    for _, r in s.iloc[b * 40:(b + 1) * 40].iterrows():
        prev = tr(r["prev_body"], 600) or "(none - this is the first post of the issue)"
        blocos.append(f'### Item {r["item"]}\nRepository: {r["repo"]}\nIssue title: {tr(r["issue_title"], 200)}\n'
                      f'Previous comment:\n"""\n{prev}\n"""\nTARGET COMMENT:\n"""\n{tr(r["comment_body"], 1500)}\n"""\n')
    (out / f"lote_{b + 1:02d}.txt").write_text("\n".join(blocos), encoding="utf-8")
print("5 lotes gerados em", out)
