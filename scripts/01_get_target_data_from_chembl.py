import os
import shutil

root = os.path.dirname(os.path.abspath(__file__))

output_dir = os.path.join(root, '..', 'processed', 'datasets', 'targets')
all_molecules_file = os.path.join(root, '..', 'processed', 'all_molecules.csv')

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

if not os.path.exists(all_molecules_file):
    raise Exception("Please run 00_get_assay_data_from_chembl.py first!")

