import os
import re
import shutil
import psycopg2
import pandas as pd
from tqdm import tqdm
import collections
from standardiser import standardise
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))

output_dir = os.path.join(root, '..', 'processed', 'datasets', 'targets')
all_molecules_file = os.path.join(root, '..', 'processed', 'all_molecules.csv')

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

DB_NAME = 'chembl_35'
DB_USER = "chembl_user"
DB_PWD = "aaa"
DB_HOST = 'localhost'
DB_PORT = 5432

MIN_TARGET_SIZE = 30
MIN_MINORITY_CLASS_SIZE = 10
PCHEMBL_CUTS = [5, 6, 7, 8]
PERCENTILE_CUTS = [1, 10, 25, 50]

if not os.path.exists(all_molecules_file):
    raise Exception("Please run 00_get_assay_data_from_chembl.py first!")

directions = {}
for v in pd.read_csv(os.path.join(root, '..', 'data', 'selected_standard_type_directions.csv'), sep=";")[["standard_type", "standard_units", "activity_direction"]].values:
    directions[(v[0], v[1])] = v[2]

def format_string(string: str) -> str:
    string = string.lower()
    string = re.sub(r'[\/:*?"<>|]', "", string)
    string = string.replace(" ", "_")
    return string

def update_all_molecules_file(ik_smiles):
    if not os.path.exists(all_molecules_file):
        df = pd.DataFrame(ik_smiles, columns=['inchikey', 'smiles'])
    else:
        df = pd.read_csv(all_molecules_file)
        df = pd.concat([df, pd.DataFrame(ik_smiles, columns=['inchikey', 'smiles'])], axis=0)
    df.drop_duplicates(subset=['inchikey'], inplace=True)
    df = df.sort_values(by='inchikey')
    df.to_csv(all_molecules_file, index=False)

def get_targets():
    conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PWD, host=DB_HOST, port=DB_PORT)
    cursor = conn.cursor()
    query_string = '''
    SELECT tid, chembl_id
    FROM target_dictionary
    '''
    cursor.execute(query_string)
    col_names = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(cursor.fetchall(), columns=col_names)
    return df

def get_dataframe_for_target(target_chembl_id):
    conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PWD, host=DB_HOST, port=DB_PORT)
    cursor = conn.cursor()
    query_string = '''
    SELECT
        ta.tid AS target_id,
        a.assay_id,
        a.assay_type,
        act.standard_type AS standard_type,
        act.standard_value AS standard_value,
        act.standard_units AS standard_unit,
        act.standard_relation AS standard_relation,
        act.pchembl_value AS pchembl_value,
        mol.canonical_smiles AS smiles,
        mol.standard_inchi_key AS inchikey
    FROM
        activities AS act
    JOIN assays AS a ON act.assay_id = a.assay_id
    JOIN target_dictionary AS ta ON a.tid = ta.tid
    JOIN compound_structures AS mol ON act.molregno = mol.molregno
    WHERE
        ta.chembl_id = '{0}' AND
        act.standard_relation = '=' AND
        act.standard_value IS NOT NULL;
    '''.format(target_chembl_id)
    cursor.execute(query_string)
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
    # Creating the datasets
    datasets = {}
    units = df[["assay_type", "standard_unit", "standard_type"]].drop_duplicates()
    for i, r in units.iterrows():
        ass_type = r["assay_type"]
        std_unit = r["standard_unit"]
        std_type = r["standard_type"]
        std_unit = units["standard_unit"].values[0]
        std_type = units["standard_type"].values[0]
        df_ = df[df["standard_unit"] == std_unit]
        df_ = df_[df_["standard_type"] == std_type]
        df_ = df_[df_["assay_type"] == ass_type]
        df_ = df_[["smiles", "inchikey", "standard_value"]].rename(columns={"standard_value": "value"})
        df_ = df_[df_["value"].notnull()]
        df_ = _dedupe_with_std(df_, ik2std, False)
        if df_.shape[0] < MIN_TARGET_SIZE:
            continue
        datasets["standard_value-type_{0}-{1}_{2}-regression".format(format_string(ass_type), format_string(std_type), format_string(std_unit))] = df_.copy()
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
                    datasets["standard_value-type_{0}-{1}_{2}-classification_{3}".format(format_string(ass_type), format_string(std_type), format_string(std_unit), cut)] = df_.copy()
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
                    datasets["standard_value-type_{0}-{1}_{2}-classification_{3}".format(format_string(ass_type), format_string(std_type), format_string(std_unit), cut)] = df_.copy()
            else:
                pass
    # do globally with pchembl_value
    df_ = df[["smiles", "inchikey", "pchembl_value"]].rename(columns={"pchembl_value": "value"})
    df_ = df_[df_["value"].notnull()]
    if df_.shape[0] >= MIN_TARGET_SIZE:
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
    datasets = {k: v for k, v in datasets.items() if v.shape[0] > MIN_TARGET_SIZE}
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

def write_datasets(datasets, target_chembl_id):
    for name, df in datasets.items():
        df.to_csv(os.path.join(output_dir, f"target_{target_chembl_id}-{name}.csv"), index=False)

dt = get_targets()
# dt = pd.DataFrame({'chembl_id': ['CHEMBL2074']})

for i, r in dt.iterrows():
    print(i)
    print(r)
    target_chembl_id = r['chembl_id']
    print(target_chembl_id)
    df = get_dataframe_for_target(target_chembl_id)
    if df.shape[0] < MIN_TARGET_SIZE:
        continue
    tasks = define_tasks(df)
    write_datasets(tasks, target_chembl_id)
