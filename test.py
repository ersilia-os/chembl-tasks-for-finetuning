import h5py

with h5py.File("processed/descriptors/datamol_descriptors.h5", "r") as f:
    print(f.keys())
    features = f["X"][:]
    print(features)