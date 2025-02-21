import h5py
import os
from tqdm import tqdm


class ChemblTasks(object):

    def __init__(self, number_of_tasks: int = 10):
        self.number_of_tasks = number_of_tasks

    def iterate(self):
        for i in range(self.number_of_tasks):
            yield i