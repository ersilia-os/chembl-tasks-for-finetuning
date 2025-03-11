import os
import pandas as pd
import h5py
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm
import numpy as np
from rdkit.Chem import Descriptors as RdkitDescriptors
from rdkit import Chem
from rdkit.Chem import MACCSkeys

root = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(root, '..', 'processed', 'descriptors')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

data_dir = os.path.join(root, '..', 'data')

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


# RDKIT descriptors

RDKIT_PROPS = {"1.0.0": ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n',
                         'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
                         'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2',
                         'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6',
                         'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'ExactMolWt',
                         'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3',
                         'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 'HeavyAtomMolWt',
                         'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'MaxAbsEStateIndex',
                         'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge',
                         'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex',
                         'MinPartialCharge', 'MolLogP', 'MolMR', 'MolWt', 'NHOHCount',
                         'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
                         'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
                         'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
                         'NumRadicalElectrons', 'NumRotatableBonds', 'NumSaturatedCarbocycles',
                         'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumValenceElectrons',
                         'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13',
                         'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5',
                         'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'RingCount',
                         'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5',
                         'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10',
                         'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4',
                         'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9',
                         'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3',
                         'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8',
                         'VSA_EState9', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
                         'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2',
                         'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0',
                         'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
                         'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide',
                         'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl',
                         'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
                         'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester',
                         'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
                         'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
                         'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine',
                         'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho',
                         'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
                         'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine',
                         'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
                         'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole',
                         'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'qed']
               }

CURRENT_VERSION = "1.0.0"


class _RdkitDescriptor(object):
    def __init__(self):
        self.properties = RDKIT_PROPS[CURRENT_VERSION]
        self._funcs = {name: func for name, func in RdkitDescriptors.descList}

    def calc(self, mols):
        R = []
        for mol in mols:
            if mol is None:
                r = [np.nan]*len(self.properties)
            else:
                r = []
                for prop in self.properties:
                    r += [self._funcs[prop](mol)]
            R += [r]
        return np.array(R)


def rdkit_featurizer(smiles):
    d = _RdkitDescriptor()
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    X = d.calc(mols)
    data = pd.DataFrame(X, columns=d.properties)
    return data


class RdkitFeaturizer(object):

    def __init__(self):
        self.batch_size = BATCH_SIZE

    def calculate(self, smiles_list):
        return rdkit_featurizer(smiles_list)
    
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
                dx = self.calculate(smiles_list)
                columns = list(dx.columns)
                X = dx.values.astype('float32')
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
                        data=np.array(columns, dtype="S"),
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


# MACCS keys

def maccs_featurizer(smiles):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    mk = os.path.join(data_dir, "MACCSkeys.txt")
    with open(str(mk), "r") as f:
        names = tuple([x.strip().split("\t")[-1] for x in f.readlines()[1:]])
    R = []
    for mol in mols:
        maccs_fp = [int(x) for x in MACCSkeys.GenMACCSKeys(mol).ToBitString()[1:]]
        R += [maccs_fp]
    return pd.DataFrame(R, columns=names)


class MaccsFeaturizer(object):

    def __init__(self):
        self.batch_size = BATCH_SIZE

    def calculate(self, smiles_list):
        return maccs_featurizer(smiles_list)

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
                dx = self.calculate(smiles_list)
                columns = list(dx.columns)
                X = dx.values.astype('int8')
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
                        data=np.array(columns, dtype="S"),
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


if __name__ == "__main__":
    print("Calculating Morgan count fingerprints 2048")
    featurizer = MorganCountFeaturizer(n_bits=2048, radius=3)
    featurizer.calculate_csv2h5(all_molecules_file, os.path.join(output_dir, "morgan_count_2048.h5"), chunksize=BATCH_SIZE)
    print("Calculating Rdkit 200 descriptors")
    featurizer = RdkitFeaturizer()
    featurizer.calculate_csv2h5(all_molecules_file, os.path.join(output_dir, "rdkit_descriptors_200.h5"), chunksize=BATCH_SIZE)
    print("Calculating MACCS featurizer")
    featurizer = MaccsFeaturizer()
    featurizer.calculate_csv2h5(all_molecules_file, os.path.join(output_dir, "maccs_keys_166.h5"), chunksize=BATCH_SIZE)