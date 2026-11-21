"""Análise do estudo piloto: compara as respostas do LLM com as anotações de Ehsani et al. (2024).

Uso: python analisar_piloto.py <pasta_com_respostas> <amostra_piloto_GABARITO.csv> <annotated_issue_level.csv>
A pasta de respostas deve ter um .txt por lote, com uma linha JSON por item.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (classification_report, cohen_kappa_score,
                             confusion_matrix, precision_recall_fscore_support)

resp_dir, gabarito_csv, issue_csv = sys.argv[1], sys.argv[2], sys.argv[3]
TBDF = ["None", "Bitter frustration", "Impatience", "Mocking", "Irony", "Vulgarity",
        "Threat", "Entitlement", "Insulting", "Identity attacks/Name-Calling"]

# 1. Ler respostas (tolerante a texto extra ou cercas de código em volta do JSON)
rows, problemas = [], []
for f in sorted(Path(resp_dir).glob("*.txt")):
    for line in f.read_text(encoding="utf-8").splitlines():
        m = re.search(r"\{.*\}", line)
        if not m:
            continue
        try:
            rows.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            problemas.append((f.name, line[:80]))
llm = pd.DataFrame(rows).drop_duplicates("item", keep="last")
gab = pd.read_csv(gabarito_csv)
gab["tbdf_ref"] = gab.tbdf_ref.fillna("None")  # o pandas lê o texto "None" como vazio
if llm.empty:
    sys.exit("Nenhuma resposta JSON encontrada na pasta indicada.")
df = gab.merge(llm, on="item", how="left")
faltando = df[df.tbdf.isna()].item.tolist()
df = df[df.tbdf.notna()].copy()
df["tbdf"] = df.tbdf.where(df.tbdf.isin(TBDF), "Invalid")
df["violation_llm"] = df.violation.astype(str).str.lower().eq("true").astype(int)

print(f"Itens avaliados: {len(df)} de {len(gab)}. Faltando: {faltando}. Linhas inválidas: {len(problemas)}")

# 2. Tarefa A: TBDF (categoria única)
print("\n=== Tarefa A: característica de incivilidade (TBDF) ===")
print(f"Acurácia: {(df.tbdf == df.tbdf_ref).mean():.3f}")
print(f"Kappa de Cohen (LLM x dataset): {cohen_kappa_score(df.tbdf_ref, df.tbdf):.3f}")
print(classification_report(df.tbdf_ref, df.tbdf, labels=TBDF, zero_division=0, digits=3))
cm = pd.DataFrame(confusion_matrix(df.tbdf_ref, df.tbdf, labels=TBDF), index=TBDF, columns=TBDF)
cm.to_csv("matriz_confusao_tbdf.csv")

# Binário: incivil x civil
inc_ref = (df.tbdf_ref != "None").astype(int)
inc_llm = (df.tbdf != "None").astype(int)
p, r, f1, _ = precision_recall_fscore_support(inc_ref, inc_llm, average="binary", zero_division=0)
print(f"Incivil x civil -> P={p:.3f} R={r:.3f} F1={f1:.3f} kappa={cohen_kappa_score(inc_ref, inc_llm):.3f}")

# 3. Tarefa B: violação de CoC contra o mapeamento a priori TBDF -> violação
print("\n=== Tarefa B: violação de CoC (referência = mapeamento a priori) ===")
p, r, f1, _ = precision_recall_fscore_support(df.violation_ref, df.violation_llm, average="binary", zero_division=0)
print(f"P={p:.3f} R={r:.3f} F1={f1:.3f} kappa={cohen_kappa_score(df.violation_ref, df.violation_llm):.3f}")
print("Proporção marcada como violação pelo LLM, por TBDF de referência:")
print(df.groupby("tbdf_ref").violation_llm.agg(["mean", "count"]).round(2).sort_values("mean"))

# 4. Confiança do modelo x acerto
if "confidence" in df:
    df["acerto_tbdf"] = (df.tbdf == df.tbdf_ref).astype(int)
    print("\nAcerto por nível de confiança:")
    print(df.groupby("confidence").acerto_tbdf.agg(["mean", "count"]).round(2))

# 5. RQ1 (sem LLM): consequências anotadas nas 404 threads trancadas
iss = pd.read_csv(issue_csv)
iss.columns = [c.strip() for c in iss.columns]
cons = iss.consequences.fillna("[]").apply(lambda s: eval(s) if s.startswith("[") else [])
n = len(iss)
print("\n=== RQ1: consequências nas threads trancadas (todo o dataset) ===")
for k in ["Invoke Code of Conduct", "Trying to stop the incivility", "Discontinued further discussion", "Escalating further"]:
    k_n = cons.apply(lambda l: k in l).sum()
    print(f"{k}: {k_n} de {n} threads ({k_n / n:.1%})")

# 6. Discordâncias para análise de erros
cols = ["item", "comment_id", "repo", "tbdf_ref", "tbdf", "violation_ref", "violation_llm", "confidence", "rationale", "comment_body"]
df[df.tbdf != df.tbdf_ref][cols].to_csv("discordancias.csv", index=False)
print("\nArquivos gerados: matriz_confusao_tbdf.csv, discordancias.csv")
