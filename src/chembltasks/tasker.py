import h5py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


root = os.path.dirname(os.path.abspath(__file__))

MIN_BASELINE_PERFORMANCE_CLASSIFICATION = 0.6
MIN_BASELINE_PERFORMANCE_REGRESSION = 0.2


class ChemblTasks(object):

    def __init__(self, processed_data_dir: str = None, tasks_per_target: int = 10, num_descriptors: int = 1, round: int = 0):
        """
        Initialize the Tasker class.
        Parameters
        ----------
        processed_data_dir : str, optional
            Directory where the processed data is stored. If None, a default directory is used.
        tasks_per_target : int, optional
            Number of tasks per target. Must be one of [1, 10, 100]. Default is 10.
        num_descriptors : int, optional
            Number of descriptors. Must be either 1 or 3. Default is 1.
        round : int, optional
            Round number. Must be in [0, 1, 2]. Default is 0.
        Raises
        ------
        AssertionError
            If `tasks_per_target` is not in [1, 10, 100].
        AssertionError
            If `num_descriptors` is not in [1, 3].
        AssertionError
            If `round` is not in range(3).
        Notes
        -----
        This method loads tasks, descriptors, and modelability data, and filters tasks based on baseline performance.
        """
        assert tasks_per_target in [1, 10, 100], "tasks_per_target must be 1, 10 or 100"
        assert num_descriptors in [1, 3], "num_descriptors must be 1 or 3"
        assert round in range(3), "round must be in [0, 1, 2]"
        if processed_data_dir is None:
            processed_data_dir = os.path.join(root, "..", "..", "processed")
        self.processed_data_dir = os.path.abspath(processed_data_dir)
        self.tasks_per_target = tasks_per_target
        self.num_descriptors = num_descriptors
        self.round = round
        print(f"Loading tasks with {tasks_per_target} tasks per target, {num_descriptors} descriptors and round {round}")
        file_name = f"tasks_n_{self.tasks_per_target}_d_{self.num_descriptors}_r_{self.round}.csv"
        file_path = os.path.join(self.processed_data_dir, "tasks", file_name)
        self.data = pd.read_csv(file_path)
        print("Loading descriptors")
        descriptors = sorted(set(self.data["descriptor"].tolist()))
        X_dict = {}
        ik2idx_dict = {}
        for d in tqdm(descriptors):
            ik2idx = {}
            with h5py.File(os.path.join(self.processed_data_dir, "descriptors", f"{d}.h5"), "r") as f:
                X_dict[d] = f["X"][:]
                identifiers = [x.decode() for x in f["identifier"][:]]
                ik2idx = {ik: i for i, ik in enumerate(identifiers)}
            ik2idx_dict[d] = ik2idx
        self.ik2idx_dict = ik2idx_dict
        self.X_dict = X_dict
        print("Loading modelability (baseline performance) data")
        modelability_csv = os.path.join(self.processed_data_dir, "modelability.csv")
        modelability = pd.read_csv(modelability_csv)
        print("Filtering tasks based on baseline performance")
        self.accepted_tasks = []
        for _, r in modelability.iterrows():
            task, task_type, performance = r["dataset"], r["task_type"], r["performance"]
            if task_type == "classification":
                if performance >= MIN_BASELINE_PERFORMANCE_CLASSIFICATION:
                    self.accepted_tasks += [task]
            else:
                if performance >= MIN_BASELINE_PERFORMANCE_REGRESSION:
                    self.accepted_tasks += [task]
        self.accepted_tasks = set(self.accepted_tasks)

    def iterate(self, split: str):
        """
        Iterate over the dataset split and yield task-specific data.

        Parameters
        ----------
        split : str
            The dataset split to iterate over. Must be one of ["train", "valid", "test"].

        Yields
        ------
        dict
            A dictionary containing the following keys:
            - "X" : np.ndarray
                The feature matrix for the task.
            - "y" : np.ndarray
                The target values for the task.
            - "task_type" : str
                The type of task, either "classification" or "regression".
            - "task" : str
                The name of the task.
        """
        assert split in ["train", "valid", "test"], "split must be in ['train', 'valid', 'test']"
        df = self.data[self.data["split"] == split]
        values = []
        for _, r in df.iterrows():
            task, descriptor = r["task"], r["descriptor"]
            if task not in self.accepted_tasks:
                continue
            values += [(task, descriptor)]
        for task, descriptor in tqdm(values):
            X = self.X_dict[descriptor]
            ik2idx = self.ik2idx_dict[descriptor]
            ds_type = task.split("_")[0]
            dt = pd.read_csv(os.path.join(self.processed_data_dir, "datasets", ds_type+"s", f"{task}.csv"))
            inchikeys = list(dt["inchikey"])
            idxs = [ik2idx[ik] for ik in inchikeys]
            y = list(dt["value"])
            if "-classification" in task:
                task_type = "classification"
                X = X[idxs]
                y = np.array(y, dtype="int8")
            else:
                task_type = "regression"
                X = X[idxs]
                y = np.array(y, dtype="float32")
            result = {"X": X, "y": y, "task_type": task_type, "task": task}
            yield result