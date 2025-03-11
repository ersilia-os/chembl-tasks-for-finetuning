import os
from chembltasks import ChemblTasks
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))

R = []
for round in [0, 1, 2]:
    for tasks_per_target in [1, 10, 100]:
        for num_descriptors in [1, 3]:
            ct = ChemblTasks(tasks_per_target=tasks_per_target, num_descriptors=num_descriptors, round=round)
            r = []
            for split in ["train", "valid", "test"]:
                n = 0
                for _ in ct.iterate(split=split):
                    n += 1
                r += [n]
            R += [[round, tasks_per_target, num_descriptors] + r]

df = pd.DataFrame(R, columns=["round", "tasks_per_target", "num_descriptors", "train", "valid", "test"])
df.to_csv(os.path.join(root, "..", "processed", "iterator_statistics.csv"), index=False)