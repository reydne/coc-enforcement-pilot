"""Supervised baselines (TF-IDF + classical models) trained on the rest of the dataset by Ehsani et al.
and evaluated on the same 200 pilot comments.

Usage: python supervised_baselines.py <annotated_comment_level.csv> <pilot_sample.csv>
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline, make_union
from sklearn.tree import DecisionTreeClassifier

data_csv, sample_csv = sys.argv[1], sys.argv[2]
c = pd.read_csv(data_csv)
c["tbdf"] = c.tbdf.replace({"Identity attack/Name-Calling": "Identity attacks/Name-Calling",
                            "Identity Attack/Name-Calling": "Identity attacks/Name-Calling"}).fillna("None")
c["comment_body"] = c.comment_body.fillna("")
s = pd.read_csv(sample_csv)
s["tbdf_ref"] = s.tbdf_ref.fillna("None")
FEWSHOT = [1229192711, 1088264852, 378762344, 1082171169, 1160417956, 1102331681]
train = c[~c.comment_id.isin(set(s.comment_id) | set(FEWSHOT))]
VIOLATION = {"Insulting", "Mocking", "Irony", "Identity attacks/Name-Calling", "Vulgarity"}
train = train.assign(unc=(train.tbdf != "None").astype(int), viol=train.tbdf.isin(VIOLATION).astype(int))
test = s.assign(unc=(s.tbdf_ref != "None").astype(int))
print(f"Training: {len(train)} comments; test: {len(test)}")


def features():
    return make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                      TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True))


models = {
    "Logistic regression": lambda: LogisticRegression(max_iter=3000, class_weight="balanced"),
    "Decision tree": lambda: DecisionTreeClassifier(class_weight="balanced", max_depth=30, random_state=42),
    "Random forest": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
}
rows = []
for name, make_model in models.items():
    for task, y_train, y_test in [("Civil vs. uncivil", train.unc, test.unc),
                                  ("Feature (TBDF)", train.tbdf, test.tbdf_ref),
                                  ("Violation (mapping)", train.viol, test.violation_ref)]:
        m = make_pipeline(features(), make_model()).fit(train.comment_body, y_train)
        pred = m.predict(test.comment_body.fillna(""))
        if task == "Feature (TBDF)":
            rows.append([name, task, accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro"),
                         cohen_kappa_score(y_test, pred), np.nan, np.nan])
        else:
            P, R, F, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
            rows.append([name, task, accuracy_score(y_test, pred), F, cohen_kappa_score(y_test, pred), P, R])
res = pd.DataFrame(rows, columns=["model", "task", "accuracy", "F1 (macro for TBDF)", "kappa", "precision", "recall"])
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
res.to_csv("baseline_results.csv", index=False)
