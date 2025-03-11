import os
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

root = os.path.dirname(os.path.abspath(__file__))

datasets_dir = os.path.abspath(os.path.join(root, '..', 'processed', 'datasets'))
modelability_csv = os.path.abspath(os.path.join(root, '..', 'processed', 'modelability.csv'))

if not os.path.exists(modelability_csv):
    df = None
else:
    df = pd.read_csv(modelability_csv)

print("Reading fingerprints...")
with h5py.File(os.path.join(root, "..", "processed", "descriptors", "morgan_count_2048.h5"), "r") as f:
    X = f["X"][:]
    identifiers = [x.decode() for x in f["identifier"][:]]
    ik2idx = {ik: i for i, ik in enumerate(identifiers)}
print("Done!")

def classification_modelability(df):
    inchikeys = list(df['inchikey'])
    y_ = np.array(df['value'], dtype=int)
    idxs = [ik2idx[ik] for ik in inchikeys]
    X_ = X[idxs]
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    aurocs = []
    for train, test in tqdm(skf.split(X_, y_)):
        clf = RandomForestClassifier(n_estimators=10, n_jobs=-1)
        clf.fit(X_[train], y_[train])
        aurocs += [roc_auc_score(y_[test], clf.predict_proba(X_[test])[:, 1])]
    return np.mean(aurocs)

def regression_modelability(df):
    inchikeys = list(df['inchikey'])
    y_ = np.array(df['value'], dtype=float)
    idxs = [ik2idx[ik] for ik in inchikeys]
    X_ = X[idxs]
    kf = KFold(n_splits=5, shuffle=True)
    r2s = []
    for train, test in tqdm(kf.split(X_)):
        reg = RandomForestRegressor(n_estimators=10, n_jobs=-1)
        reg.fit(X_[train], y_[train])
        r2s += [pearsonr(y_[test], reg.predict(X_[test]))[0]]
    return np.mean(r2s)

for ds_type  in ["assays", "targets"]:
    for fn in tqdm(os.listdir(os.path.join(datasets_dir, ds_type))):
        if not fn.endswith(".csv"):
            continue
        name = fn[:-4]
        if df is None:
            do_estimation = True
        else:
            done_names = df["dataset"].tolist()
            if name in done_names:
                do_estimation = False
            else:
                do_estimation = True
        if not do_estimation:
            continue
        if "-classification" in name:
            task_type = "classification"
            performance_metric = "auroc"
        elif "-regression" in name:
            task_type = "regression"
            performance_metric = "rho"
        else:
            raise ValueError("Unknown task type")
        dt = pd.read_csv(os.path.join(datasets_dir, ds_type, fn))
        num_samples = dt.shape[0]
        if task_type == "classification":
            performance = classification_modelability(dt)
        else:
            performance = regression_modelability(dt)
        data = {"dataset_type": [ds_type], "dataset": [name], "task_type": [task_type], "performance": [performance], "performance_metric": [performance_metric], "num_samples": [num_samples]}
        df_ = pd.DataFrame(data)
        if df is None:
            df = df_
        else:
            df = pd.concat([df, df_])
        df.to_csv(modelability_csv, index=False)
        