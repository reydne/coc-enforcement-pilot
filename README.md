# Replication package: pilot study

This package contains everything needed to reproduce the pilot study, which evaluates an LLM as a classifier of incivility and code of conduct (CoC) violations in GitHub comments.

## Source data

The pilot uses the dataset by Ehsani et al., *Incivility in Open Source Projects: A Comprehensive Annotated Dataset of Locked GitHub Issue Threads* (MSR 2024), available at https://github.com/vcu-swim-lab/incivility-dataset (MIT License, copy in `1_dados/`). The sample in this package redistributes comment texts from that dataset under the same license.

## Structure

| Folder | Content |
|---|---|
| `1_dados/` | `amostra_piloto.csv`: the 200 sampled comments, with the dataset annotation (`tbdf_ref`) and the a priori violation mapping (`violation_ref`); `fewshot_usados.csv`: the six few-shot examples; dataset license |
| `2_prompt/` | `cabecalho_prompt_v1.txt`: instructions, taxonomy and few-shot examples; `lotes/`: the five batches of 40 comments sent to the model (no reference labels) |
| `3_respostas_llm/` | Raw model outputs, one JSON object per comment |
| `4_scripts/` | Sampling, batch generation, analysis and supervised baselines |
| `5_resultados/` | Metrics, confusion matrix, disagreements and baseline results |

## Procedure

1. **Sample.** `preparar_amostra.py` draws a stratified sample with fixed seeds: 100 uncivil comments across the nine incivility features (TBDFs) and 100 civil comments. Issues containing the few-shot examples are excluded. Violations are defined a priori: insulting, mocking, irony, identity attacks and vulgarity are mapped to violations; bitter frustration, impatience, entitlement and threat are not.
2. **Batches.** `gerar_lotes.py` builds five batches of 40 comments. Each item contains the repository, the issue title, the preceding comment (truncated to 600 characters) and the target comment (truncated to 1,500 characters). The model never sees the reference labels or the lock reason.
3. **Classification.** Each batch was submitted to Claude Sonnet 5.5 through the claude.ai interface, in a new conversation, on 20 November 2026, with the message: `Follow the instructions in the attached file and classify all 40 items.` Sampling parameters could not be controlled in this interface.
4. **Analysis.** `analisar_piloto.py` compares the model output with the reference labels. `baseline_supervisionado.py` trains TF-IDF models (logistic regression, decision tree, random forest) on the remaining annotated comments of the dataset and evaluates them on the same 200 comments.

## Reproducing

```bash
pip install -r requirements.txt
git clone https://github.com/vcu-swim-lab/incivility-dataset
python 4_scripts/preparar_amostra.py incivility-dataset/dataset 1_dados
python 4_scripts/gerar_lotes.py 1_dados/amostra_piloto.csv 2_prompt/cabecalho_prompt_v1.txt 2_prompt/lotes
cd 5_resultados
python ../4_scripts/analisar_piloto.py ../3_respostas_llm ../1_dados/amostra_piloto.csv ../incivility-dataset/dataset/annotated_issue_level.csv
python ../4_scripts/baseline_supervisionado.py ../incivility-dataset/dataset/annotated_comment_level.csv ../1_dados/amostra_piloto.csv
```

The first two commands regenerate exactly the sample and the batches included in this package. Running the classification again may yield different outputs, because LLM responses are not deterministic.

## Main results

| Task | Metric | LLM | Best baseline (logistic regression) |
|---|---|---|---|
| Civil vs. uncivil | F1 / kappa | 0.78 / 0.58 | 0.71 / 0.50 |
| Incivility feature (TBDF) | macro F1 / kappa | 0.38 / 0.40 | 0.30 / 0.37 |
| CoC violation (a priori mapping) | F1 / kappa | 0.40 / 0.28 | 0.52 / 0.41 |

Only 35 of the 404 locked threads (8.7%) contain an explicit invocation of the CoC, according to the dataset annotations.
