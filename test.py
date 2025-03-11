from chembltasks import ChemblTasks
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from tqdm import tqdm

def classification_modelability(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    aurocs = []
    for train, test in tqdm(skf.split(X, y)):
        clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
        clf.fit(X[train], y[train])
        aurocs += [roc_auc_score(y[test], clf.predict_proba(X[test])[:, 1])]
    return np.mean(aurocs)

def regression_modelability(X, y):
    kf = KFold(n_splits=5, shuffle=True)
    r2s = []
    for train, test in tqdm(kf.split(X)):
        reg = RandomForestRegressor(n_estimators=10, n_jobs=-1)
        reg.fit(X[train], y[train])
        r2s += [pearsonr(y[test], reg.predict(X[test]))[0]]
    return np.mean(r2s)

tasks_per_target = 10
num_descriptors = 3
round = 0
ct = ChemblTasks(tasks_per_target=tasks_per_target, num_descriptors=num_descriptors, round=round)

print("train")
for r in ct.iterate("train"):
    if r["X"].shape[1] < 300:
        print(r["X"].shape)
        print(r["X"])
        print(r["y"])
    else:
        continue
    if r["task_type"] == "classification":
        performance = classification_modelability(r["X"], r["y"])
    else:
        performance = regression_modelability(r["X"], r["y"])
    print(r["task_type"], performance)

print("valid")
for r in ct.iterate("valid"):
    continue

print("test")
for r in ct.iterate("test"):
    continue