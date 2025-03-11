import os
import pandas as pd
import psycopg2

DB_NAME = 'chembl_35'
DB_USER = "chembl_user"
DB_PWD = "aaa"
DB_HOST = 'localhost'
DB_PORT = 5432

query = '''
SELECT DISTINCT
    ta.chembl_id AS target_id,
    a.chembl_id AS assay_id
FROM
    activities AS act
JOIN assays AS a ON act.assay_id = a.assay_id
JOIN target_dictionary AS ta ON a.tid = ta.tid;'
'''

root = os.path.dirname(os.path.abspath(__file__))

def get_assay_target_pairs():
    conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PWD, host=DB_HOST, port=DB_PORT)
    cursor = conn.cursor()
    query_string = '''
    SELECT DISTINCT
        ta.chembl_id AS target_id,
        a.chembl_id AS assay_id
    FROM
        activities AS act
    JOIN assays AS a ON act.assay_id = a.assay_id
    JOIN target_dictionary AS ta ON a.tid = ta.tid;
    '''
    cursor.execute(query_string)
    col_names = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(cursor.fetchall(), columns=col_names)
    return df

df = get_assay_target_pairs()
df.to_csv(os.path.join(root, "..", "processed", "assay_has_target.csv"), index=False)

