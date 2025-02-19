import os
import shutil
import psycopg2
import pandas as pd
from tqdm import tqdm

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

MIN_ASSAY_SIZE = 30
MIN_MINORITY_CLASS_SIZE = 10
PCHEMBL_CUTS = [5, 6, 7, 8]
PERCENTILE_CUTS = [1, 10, 25, 50]

if not os.path.exists(all_molecules_file):
    raise Exception("Please run 00_get_assay_data_from_chembl.py first!")

directions = {}
for v in pd.read_csv(os.path.join(root, '..', 'data', 'selected_standard_type_directions.csv'), sep=";")[["standard_type", "standard_units", "activity_direction"]].values:
    directions[(v[0], v[1])] = v[2]

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

def get_dataframe_for_target(tid):
    pass



dt = get_targets()

