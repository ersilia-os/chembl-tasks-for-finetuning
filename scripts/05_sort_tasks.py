import os
import csv
import collections
import random
from tqdm import tqdm
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))

datasets_dir = os.path.abspath(os.path.join(root, '..', 'processed', 'datasets'))

if not os.path.exists(os.path.join(root, "..", "processed", "tasks")):
    os.makedirs(os.path.join(root, "..", "processed", "tasks"))


def main(round):

    all_assays = []
    for fn in os.listdir(os.path.join(datasets_dir, "assays")):
        if not fn.endswith(".csv"):
            continue
        all_assays += [fn.split("-")[0].split("_")[1]]

    all_targets = []
    for fn in os.listdir(os.path.join(datasets_dir, "targets")):
        if not fn.endswith(".csv"):
            continue
        all_targets += [fn.split("-")[0].split("_")[1]]

    all_assays = sorted(set(all_assays))
    all_targets = sorted(set(all_targets))

    target2assays = collections.defaultdict(list)
    with open(os.path.join(root, "..", "processed", "assay_has_target.csv"), "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            target2assays[row[0]] += [row[1]]

    random.shuffle(all_targets)
    random.shuffle(all_assays)

    split_index_1 = int(0.8 * len(all_targets))  # First 80%
    split_index_2 = int(0.9 * len(all_targets))  # Next 10%

    train_targets = all_targets[:split_index_1]   # 80%
    valid_targets = all_targets[split_index_1:split_index_2]  # 10%
    test_targets = all_targets[split_index_2:]   # 10%

    train_assays = [a for t in train_targets for a in target2assays[t]]
    valid_assays = [a for t in valid_targets for a in target2assays[t]]
    test_assays = [a for t in test_targets for a in target2assays[t]]

    target2task = collections.defaultdict(list)
    for fn in os.listdir(os.path.join(root, "..", "processed", "datasets", "targets")):
        if not fn.endswith(".csv"):
            continue
        name = fn[:-4]
        target2task[name.split("-")[0].split("_")[1]] += [name]

    assay2task = collections.defaultdict(list)
    for fn in os.listdir(os.path.join(root, "..", "processed", "datasets", "assays")):
        if not fn.endswith(".csv"):
            continue
        name = fn[:-4]
        assay2task[name.split("-")[0].split("_")[1]] += [name]

    def tasks_per_target(max_n, descriptors, train_targets, valid_targets, test_targets):
        train_targets = train_targets[:]
        random.shuffle(train_targets)
        valid_targets = valid_targets[:]
        random.shuffle(valid_targets)
        test_targets = test_targets[:]
        random.shuffle(test_targets)
        all_targets = train_targets + valid_targets + test_targets
        all_splits = ["train"] * len(train_targets) + ["valid"] * len(valid_targets) + ["test"] * len(test_targets)
        kept_tasks = []
        for target, split in tqdm(zip(all_targets, all_splits)):
            tasks = []
            for task in target2task[target]:
                for descriptor in descriptors:
                    tasks += [(task, descriptor, split)]
            assays = target2assays[target]
            for assay in assays:
                for task in assay2task[assay]:
                    for descriptor in descriptors:
                        tasks += [[task, descriptor, split]]
            random.shuffle(tasks)
            tasks = tasks[:max_n]
            kept_tasks += tasks
        df = pd.DataFrame(kept_tasks, columns=["task", "descriptor", "split"])
        df_0 = df[df["split"] == "train"]
        df_1 = df[df["split"] == "valid"]
        df_2 = df[df["split"] == "test"]
        df_0 = df_0.sample(frac=1, random_state=42).reset_index(drop=True)
        df_1 = df_1.sample(frac=1, random_state=42).reset_index(drop=True)
        df_2 = df_2.sample(frac=1, random_state=42).reset_index(drop=True)
        df = pd.concat([df_0, df_1, df_2], axis=0)
        return df

    df = tasks_per_target(1, ["morgan_count_2048"], train_targets, valid_targets, test_targets)
    df.to_csv(os.path.join(root, "..", "processed", "tasks", f"tasks_n_1_d_1_r_{round}.csv"), index=False)

    df = tasks_per_target(1, ["morgan_count_2048", "maccs_keys_166", "rdkit_descriptors_200"], train_targets, valid_targets, test_targets)
    df.to_csv(os.path.join(root, "..", "processed", "tasks", f"tasks_n_1_d_3_r_{round}.csv"), index=False)

    df = tasks_per_target(10, ["morgan_count_2048"], train_targets, valid_targets, test_targets)
    df.to_csv(os.path.join(root, "..", "processed", "tasks", f"tasks_n_10_d_1_r_{round}.csv"), index=False)

    df = tasks_per_target(10, ["morgan_count_2048", "maccs_keys_166", "rdkit_descriptors_200"], train_targets, valid_targets, test_targets)
    df.to_csv(os.path.join(root, "..", "processed", "tasks", f"tasks_n_10_d_3_r_{round}.csv"), index=False)

    df = tasks_per_target(100, ["morgan_count_2048"], train_targets, valid_targets, test_targets)
    df.to_csv(os.path.join(root, "..", "processed", "tasks", f"tasks_n_100_d_1_r_{round}.csv"), index=False)

    df = tasks_per_target(100, ["morgan_count_2048", "maccs_keys_166", "rdkit_descriptors_200"], train_targets, valid_targets, test_targets)
    df.to_csv(os.path.join(root, "..", "processed", "tasks", f"tasks_n_100_d_3_r_{round}.csv"), index=False)


if __name__ == "__main__":
    for i in range(3):
        print("Round", i)
        main(i)