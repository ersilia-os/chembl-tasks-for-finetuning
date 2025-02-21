import os
import shutil
import psycopg2
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import collections
from standardiser import standardise
from tqdm import tqdm

root = os.path.dirname(os.path.abspath(__file__))

output_dir = os.path.join(root, '..', 'processed', 'datasets', 'assays')
all_molecules_file = os.path.join(root, '..', 'processed', 'all_molecules.csv')

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

if os.path.exists(all_molecules_file):
    os.remove(all_molecules_file)

MIN_ASSAY_SIZE = 30
MIN_MINORITY_CLASS_SIZE = 10
PCHEMBL_CUTS = [5, 6, 7, 8]
PERCENTILE_CUTS = [1, 10, 25, 50]

DB_NAME = 'chembl_35'
DB_USER = "chembl_user"
DB_PWD = "aaa"
DB_HOST = 'localhost'
DB_PORT = 5432

directions = {}
for v in pd.read_csv(os.path.join(root, '..', 'data', 'selected_standard_type_directions.csv'), sep=";")[["standard_type", "standard_units", "activity_direction"]].values:
    directions[(v[0], v[1])] = v[2]

def update_all_molecules_file(ik_smiles):
    if not os.path.exists(all_molecules_file):
        df = pd.DataFrame(ik_smiles, columns=['inchikey', 'smiles'])
    else:
        df = pd.read_csv(all_molecules_file)
        df = pd.concat([df, pd.DataFrame(ik_smiles, columns=['inchikey', 'smiles'])], axis=0)
    df.drop_duplicates(subset=['inchikey'], inplace=True)
    df = df.sort_values(by='inchikey')
    df.to_csv(all_molecules_file, index=False)

def get_assays():
    conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PWD, host=DB_HOST, port=DB_PORT)
    cursor = conn.cursor()
    query_string = '''
    SELECT
        a.assay_id, a.chembl_id
    FROM
        assays a
    JOIN
        activities act ON a.assay_id = act.assay_id
    GROUP BY
        a.assay_id, a.chembl_id
    HAVING
        COUNT(act.activity_id) >= {0}
    '''.format(MIN_ASSAY_SIZE)
    cursor.execute(query_string)
    col_names = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(cursor.fetchall(), columns=col_names)
    return df

def get_dataframe_for_assay(assay_id):
    conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PWD, host=DB_HOST, port=DB_PORT)
    cursor = conn.cursor()
    query = '''
    SELECT
        cs.canonical_smiles AS smiles,
        cs.standard_inchi_key AS inchikey,
        act.standard_value,
        act.standard_units AS standard_unit,
        act.standard_type,
        act.pchembl_value,
        act.standard_relation
    FROM
        activities act
    JOIN
        compound_structures cs ON act.molregno = cs.molregno
    WHERE
        act.assay_id = {0} AND
        act.standard_relation = '=';
    '''.format(assay_id)
    cursor.execute(query)
    col_names = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(cursor.fetchall(), columns=col_names)
    df["standard_value"] = df["standard_value"].astype(float)
    df["pchembl_value"] = df["pchembl_value"].astype(float)
    return df

def define_tasks(df):
    def _dedupe_with_std(df, ik2std, is_binary):
        values = df["value"].tolist()
        iks = df["inchikey"].tolist()
        data = collections.defaultdict(list)
        for ik, value in zip(iks, values):
            if ik in ik2std:
                data[ik2std[ik]] += [value]
        if is_binary:
            data = {k[0]: 1 if any(v) else 0 for k, v in data.items()}
        else:
            data = {k[0]: sum(v)/len(v) for k, v in data.items()}
        return pd.DataFrame(data.items(), columns=["inchikey", "value"])
    # Standardising molecules
    ik2std = {}
    for _, row in tqdm(df.iterrows(), desc="Standardising molecules", total=df.shape[0]):
        ik = row["inchikey"]
        smi = row["smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            mol = standardise.run(mol)
            if mol is None:
                continue
            wt = Descriptors.MolWt(mol)
            if wt > 1000:
                continue
            if wt < 100:
                continue
            smi = Chem.MolToSmiles(mol)
            inchikey = Chem.MolToInchiKey(mol)
            ik2std[ik] = (inchikey, smi)
        except:
            continue
    stdik2stdsmi = {v[0]: v[1] for v in ik2std.values()}
    n = df.shape[0]
    n_with_pchembl = df[df["pchembl_value"].notnull()].shape[0]
    # Creating the datasets
    datasets = {}
    df_ = df[["smiles", "inchikey", "standard_value"]].rename(columns={"standard_value": "value"})
    df_ = df_[df_["value"].notnull()]
    df_ = _dedupe_with_std(df_, ik2std, False)
    datasets["standard_value-regression"] = df_.copy()
    units = df[["standard_unit", "standard_type"]].drop_duplicates()
    if units.shape[0] == 1:
        std_unit = units["standard_unit"].values[0]
        std_type = units["standard_type"].values[0]
        if (std_type, std_unit) in directions:
            direction = directions[(std_type, std_unit)]
            if direction == 1:
                for cut in PERCENTILE_CUTS:
                    df_ = df[["smiles", "inchikey", "standard_value"]].rename(columns={"standard_value": "value"})
                    y = []
                    cut_value = np.percentile(np.array(df_["value"]), 100-cut)
                    for v in df_["value"].tolist():
                        if v >= cut_value:
                            y += [1]
                        else:
                            y += [0]
                    df_["value"] = y
                    df_ = df_[df_["value"].notnull()]
                    df_ = _dedupe_with_std(df_, ik2std, True)
                    datasets[f"standard_value-classification_{cut}"] = df_.copy()
            elif direction == -1:
                for cut in PERCENTILE_CUTS:
                    df_ = df[["smiles", "inchikey", "standard_value"]].rename(columns={"standard_value": "value"})
                    y = []
                    cut_value = np.percentile(np.array(df_["value"]), cut)
                    for v in df_["value"].tolist():
                        if v <= cut_value:
                            y += [1]
                        else:
                            y += [0]
                    df_["value"] = y
                    df_ = df_[df_["value"].notnull()]
                    df_ = _dedupe_with_std(df_, ik2std, True)
                    datasets[f"standard_value-classification_{cut}"] = df_.copy()
            else:
                pass
    if n_with_pchembl/n > 0.8:
        df_ = df[["smiles", "inchikey", "pchembl_value"]].rename(columns={"pchembl_value": "value"})
        df_ = df_[df_["value"].notnull()]
        df_ = _dedupe_with_std(df_, ik2std, False)
        datasets["pchembl_value-regression"] = df_.copy()
        for cut in PCHEMBL_CUTS:
            df_ = df[["smiles", "inchikey", "pchembl_value"]].rename(columns={"pchembl_value": "value"})
            df_ = df_[df_["value"].notnull()]
            y = []
            for v in df_["value"].tolist():
                if v >= cut:
                    y += [1]
                else:
                    y += [0]
            df_["value"] = y
            df_ = _dedupe_with_std(df_, ik2std, True)
            datasets[f"pchembl_value-classification_{cut}"] = df_.copy()
    # Filtering tasks
    datasets = {k: v for k, v in datasets.items() if v.shape[0] > MIN_ASSAY_SIZE}
    remove_datasets = []
    for name, df in datasets.items():
        if "classification" in name:
            positive_class_size = df["value"].sum()
            negative_class_size = df.shape[0] - positive_class_size
            if positive_class_size < MIN_MINORITY_CLASS_SIZE or negative_class_size < MIN_MINORITY_CLASS_SIZE:
                remove_datasets += [name]
    for name in remove_datasets:
        del datasets[name]
    # Updating all_molecules_file
    all_inchikeys = []
    for ds in datasets.values():
        all_inchikeys += ds["inchikey"].tolist()
    all_inchikeys = list(set(all_inchikeys))
    inchikey_smiles = [(ik, stdik2stdsmi[ik]) for ik in all_inchikeys]
    update_all_molecules_file(inchikey_smiles)
    return datasets

def write_datasets(datasets, assay_id):
    for name, df in datasets.items():
        df.to_csv(os.path.join(output_dir, f"assay_{assay_id}-{name}.csv"), index=False)

da = get_assays()

for i, row in tqdm(da.iterrows(), desc="Iterating over assays", total=da.shape[0]):
    print(i)
    print(row)
    assay_id = row['assay_id']
    chembl_id = row['chembl_id']
    df = get_dataframe_for_assay(assay_id)
    if df.shape[0] < MIN_ASSAY_SIZE:
        continue
    tasks = define_tasks(df)
    write_datasets(tasks, chembl_id)