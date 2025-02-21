import os
import pandas as pd
import h5py
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
import numpy as np
import datamol as dm

root = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(root, '..', 'processed', 'descriptors')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

all_molecules_file = os.path.join(root, '..', 'processed', 'all_molecules.csv')
if not os.path.exists(all_molecules_file):
    raise Exception("Please fetch molecules first!")

df = pd.read_csv(all_molecules_file)

BATCH_SIZE = 1000


class MorganCountFeaturizer(object):

    def __init__(self, n_bits, radius):
        self.batch_size = BATCH_SIZE
        self.n_bits = n_bits
        self.radius = radius

    @staticmethod
    def _clip_sparse(vect, nbits):
        l = [0]*nbits
        for i,v in vect.GetNonzeroElements().items():
            l[i] = v if v < 255 else 255
        return l

    def calculate(self, smiles_list):
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)
        X = np.zeros((len(smiles_list), self.n_bits), dtype="int8")
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            v = mfpgen.GetCountFingerprint(mol)
            X[i,:] = self._clip_sparse(v, self.n_bits)
        return X
    
    def calculate_csv2h5(self, csv_path, h5_path, chunksize=None, max_rows_ceil=None):
        if os.path.exists(h5_path):
            os.remove(h5_path)

        if chunksize is None:
            chunksize = self.batch_size
        # Define special dtype for variable-length strings
        string_dtype = h5py.special_dtype(vlen=str)
        
        with h5py.File(h5_path, "a") as f:
            identifier_col = 'inchikey'
            smiles_col = 'smiles'
            df_chunker = pd.read_csv(csv_path, chunksize=chunksize, usecols=[identifier_col, smiles_col])
            for i, chunk in tqdm(enumerate(df_chunker)):
                # Convert columns to lists of Python strings
                identifiers_list = chunk[identifier_col].astype(str).to_list()
                smiles_list = chunk[smiles_col].astype(str).to_list()
                X = self.calculate(smiles_list).astype('int8')  # Integer matrix
                if i == 0:
                    # Create datasets with variable-length string dtype
                    f.create_dataset(
                        "X",
                        data=X,
                        shape=X.shape,
                        maxshape=(None, X.shape[1]),
                        dtype='int8',  # Integer type for X
                        chunks=(chunksize, X.shape[1]),
                        compression="gzip",
                    )
                    f.create_dataset(
                        "identifier",
                        data=identifiers_list,
                        shape=(len(identifiers_list),),
                        maxshape=(None,),
                        dtype=string_dtype,  # Variable-length string
                        compression="gzip",
                    )
                    f.create_dataset(
                        "smiles",
                        data=smiles_list,
                        shape=(len(smiles_list),),
                        maxshape=(None,),
                        dtype=string_dtype,  # Variable-length string
                        compression="gzip",
                    )
                    f.create_dataset(\
                        "features",
                        data=np.array(["feature_{0}".format(str(i).zfill(4)) for i in range(X.shape[1])], dtype="S"),
                        shape=(X.shape[1],),
                        maxshape=(X.shape[1],),
                        dtype=string_dtype,
                        compression="gzip",
                    )
                else:
                    # Append to datasets
                    f["X"].resize((f["X"].shape[0] + X.shape[0]), axis=0)
                    f["X"][-X.shape[0]:] = X
                    f["identifier"].resize((f["identifier"].shape[0] + len(identifiers_list)), axis=0)
                    f["identifier"][-len(identifiers_list):] = identifiers_list
                    f["smiles"].resize((f["smiles"].shape[0] + len(smiles_list)), axis=0)
                    f["smiles"][-len(smiles_list):] = smiles_list
                if max_rows_ceil is not None and f["X"].shape[0] >= max_rows_ceil:
                    break


class DatamolFeaturizer(object):
    
    def __init__(self):
        self.batch_size = BATCH_SIZE
        self.features = None

    def calculate(self, smiles_list):
        mols = [dm.to_mol(smiles) for smiles in smiles_list]
        desc = dm.descriptors.batch_compute_many_descriptors(mols)
        if self.features is None:
            self.features = list(desc.columns)
        return np.array(desc, dtype="float32")
    
    def calculate_csv2h5(self, csv_path, h5_path, chunksize=None, max_rows_ceil=None):
        if os.path.exists(h5_path):
            os.remove(h5_path)

        if chunksize is None:
            chunksize = self.batch_size
        # Define special dtype for variable-length strings
        string_dtype = h5py.special_dtype(vlen=str)
        
        with h5py.File(h5_path, "a") as f:
            identifier_col = 'inchikey'
            smiles_col = 'smiles'
            df_chunker = pd.read_csv(csv_path, chunksize=chunksize, usecols=[identifier_col, smiles_col])
            for i, chunk in tqdm(enumerate(df_chunker)):
                # Convert columns to lists of Python strings
                identifiers_list = chunk[identifier_col].astype(str).to_list()
                smiles_list = chunk[smiles_col].astype(str).to_list()
                X = self.calculate(smiles_list).astype('float32')  # Integer matrix
                if i == 0:
                    # Create datasets with variable-length string dtype
                    f.create_dataset(
                        "X",
                        data=X,
                        shape=X.shape,
                        maxshape=(None, X.shape[1]),
                        dtype='float32',  # Integer type for X
                        chunks=(chunksize, X.shape[1]),
                        compression="gzip",
                    )
                    f.create_dataset(
                        "identifier",
                        data=identifiers_list,
                        shape=(len(identifiers_list),),
                        maxshape=(None,),
                        dtype=string_dtype,  # Variable-length string
                        compression="gzip",
                    )
                    f.create_dataset(
                        "smiles",
                        data=smiles_list,
                        shape=(len(smiles_list),),
                        maxshape=(None,),
                        dtype=string_dtype,  # Variable-length string
                        compression="gzip",
                    )
                    f.create_dataset(\
                        "features",
                        data=np.array(self.features, dtype="S"),
                        shape=(X.shape[1],),
                        maxshape=(X.shape[1],),
                        dtype=string_dtype,
                        compression="gzip",
                    )
                else:
                    # Append to datasets
                    f["X"].resize((f["X"].shape[0] + X.shape[0]), axis=0)
                    f["X"][-X.shape[0]:] = X
                    f["identifier"].resize((f["identifier"].shape[0] + len(identifiers_list)), axis=0)
                    f["identifier"][-len(identifiers_list):] = identifiers_list
                    f["smiles"].resize((f["smiles"].shape[0] + len(smiles_list)), axis=0)
                    f["smiles"][-len(smiles_list):] = smiles_list
                if max_rows_ceil is not None and f["X"].shape[0] >= max_rows_ceil:
                    break
                break

        
print("Calculating Morgan count fingerprints 2048")
featurizer = MorganCountFeaturizer(n_bits=2048, radius=3)
featurizer.calculate_csv2h5(all_molecules_file, os.path.join(output_dir, "morgan_count_2048.h5"), chunksize=BATCH_SIZE)
print("Calculating Datamol descriptors")
featurizer = DatamolFeaturizer()
featurizer.calculate_csv2h5(all_molecules_file, os.path.join(output_dir, "datamol_descriptors.h5"), chunksize=BATCH_SIZE)