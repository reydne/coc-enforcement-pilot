"""Baselines supervisionados (TF-IDF + modelos clássicos) treinados no restante do dataset de Ehsani et al.
e avaliados nos mesmos 200 comentários do piloto."""
import sys, json, glob
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support, cohen_kappa_score, accuracy_score

data_csv = sys.argv[1]  # annotated_comment_level.csv
# uso: python baseline_supervisionado.py <annotated_comment_level.csv> <amostra_piloto.csv>
c = pd.read_csv(data_csv)
c['tbdf'] = c.tbdf.replace({'Identity attack/Name-Calling': 'Identity attacks/Name-Calling',
                            'Identity Attack/Name-Calling': 'Identity attacks/Name-Calling'}).fillna('None')
c['comment_body'] = c.comment_body.fillna('')
g = pd.read_csv(sys.argv[2]); g['tbdf_ref'] = g.tbdf_ref.fillna('None')
fewshot = [1229192711, 1088264852, 378762344, 1082171169, 1160417956, 1102331681]
train = c[~c.comment_id.isin(set(g.comment_id) | set(fewshot))]
VIOL = {'Insulting', 'Mocking', 'Irony', 'Identity attacks/Name-Calling', 'Vulgarity'}
train = train.assign(inc=(train.tbdf != 'None').astype(int), viol=train.tbdf.isin(VIOL).astype(int))
test = g.assign(inc=(g.tbdf_ref != 'None').astype(int))
print(f'Treino: {len(train)} comentários; teste: {len(test)}')

def feats():
    return make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                      TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, sublinear_tf=True))
models = {
    'Regressão logística': lambda: LogisticRegression(max_iter=3000, class_weight='balanced'),
    'Árvore de decisão': lambda: DecisionTreeClassifier(class_weight='balanced', max_depth=30, random_state=42),
    'Random forest': lambda: RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1),
}
rows = []
for name, mk in models.items():
    for task, ytr, yte in [('Civil x incivil', train.inc, test.inc),
                           ('Tipo (TBDF)', train.tbdf, test.tbdf_ref),
                           ('Violação (mapeamento)', train.viol, test.violation_ref)]:
        m = make_pipeline(feats(), mk()).fit(train.comment_body, ytr)
        p = m.predict(test.comment_body.fillna(''))
        if task == 'Tipo (TBDF)':
            rows.append([name, task, accuracy_score(yte, p), f1_score(yte, p, average='macro'), cohen_kappa_score(yte, p), np.nan, np.nan])
        else:
            P, R, F, _ = precision_recall_fscore_support(yte, p, average='binary', zero_division=0)
            rows.append([name, task, accuracy_score(yte, p), F, cohen_kappa_score(yte, p), P, R])
res = pd.DataFrame(rows, columns=['modelo', 'tarefa', 'acurácia', 'F1 (macro p/ TBDF)', 'kappa', 'precisão', 'revocação'])
pd.set_option('display.width', 200)
print(res.round(3).to_string(index=False))
res.to_csv('baseline_resultados.csv', index=False)
