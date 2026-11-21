# Replication package: pilot study

This package contains everything needed to reproduce the pilot study, which evaluates a large language model (LLM) as a classifier of incivility and code of conduct (CoC) violations in GitHub comments.

## Source data

The pilot uses the dataset by Ehsani et al., *Incivility in Open Source Projects: A Comprehensive Annotated Dataset of Locked GitHub Issue Threads* (MSR 2024), available at https://github.com/vcu-swim-lab/incivility-dataset under the MIT License (a copy is in `1_data/`). The sample in this package redistributes comment texts from that dataset under the same license.

## Structure

| Folder | Content |
|---|---|
| `1_data/` | `pilot_sample.csv`: the 200 sampled comments, with the dataset annotation (`tbdf_ref`) and the a priori violation mapping (`violation_ref`); `fewshot_examples.csv`: the six few-shot examples; dataset license |
| `2_prompt/` | `prompt_header_v1.txt`: instructions, taxonomy, and few-shot examples; `batches/`: the five batches of 40 comments sent to the model, without reference labels |
| `3_llm_responses/` | Raw model outputs, one JSON object per comment |
| `4_scripts/` | Sampling, batch generation, analysis, and supervised baselines |
| `5_results/` | Metrics, confusion matrix, disagreements, and baseline results |

## Procedure

1. **Sample.** `prepare_sample.py` draws a stratified sample with fixed seeds: 100 uncivil comments across the nine incivility features (Tone Bearing Discussion Features, TBDFs) and 100 civil comments. The issues that contain the few-shot examples are excluded. Violations are defined a priori: insulting, mocking, irony, identity attacks, and vulgarity are mapped to violations; bitter frustration, impatience, entitlement, and threat are not.
2. **Batches.** `generate_batches.py` builds five batches of 40 comments. Each item contains the repository, the issue title, the preceding comment (truncated to 600 characters), and the target comment (truncated to 1,500 characters). The model never sees the reference labels or the lock reason.
3. **Classification.** Each batch was submitted to Claude Sonnet 5.5 through the claude.ai interface, in a new conversation, on 20 November 2026, with the message `Follow the instructions in the attached file and classify all 40 items.` Sampling parameters cannot be controlled in this interface.
4. **Analysis.** `analyze_pilot.py` compares the model outputs with the reference labels. `supervised_baselines.py` trains TF-IDF models (logistic regression, decision tree, and random forest) on the remaining annotated comments of the dataset and evaluates them on the same 200 comments.

## Reproducing

```bash
pip install -r requirements.txt
git clone https://github.com/vcu-swim-lab/incivility-dataset
python 4_scripts/prepare_sample.py incivility-dataset/dataset 1_data
python 4_scripts/generate_batches.py 1_data/pilot_sample.csv 2_prompt/prompt_header_v1.txt 2_prompt/batches
cd 5_results
python ../4_scripts/analyze_pilot.py ../3_llm_responses ../1_data/pilot_sample.csv ../incivility-dataset/dataset/annotated_issue_level.csv
python ../4_scripts/supervised_baselines.py ../incivility-dataset/dataset/annotated_comment_level.csv ../1_data/pilot_sample.csv
```

The first two scripts regenerate exactly the sample and the batches included in this package. Running the classification again may yield different outputs, because LLM responses are not deterministic.

## Main results

| Task | Metric | LLM | Best baseline (logistic regression) |
|---|---|---|---|
| Civil vs. uncivil | F1 / kappa | 0.78 / 0.58 | 0.71 / 0.50 |
| Incivility feature (TBDF) | macro F1 / kappa | 0.38 / 0.40 | 0.30 / 0.37 |
| CoC violation (a priori mapping) | F1 / kappa | 0.40 / 0.28 | 0.52 / 0.41 |

According to the dataset annotations, only 35 of the 404 locked threads (8.7%) contain an explicit invocation of the CoC.
