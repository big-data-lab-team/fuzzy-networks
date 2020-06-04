import pickle
import os
import glob

DATA_DIR = 'results'


class ExperimentResults:
    def __init__(self, subdir=None):
        self.dir = subdir or self.get_experiment_dir()
        if subdir is None:
            print(f'Using directory {self.dir}')

    @staticmethod
    def get_experiment_dir():
        exp_id = 1
        while True:
            exp_dir = f"exp_{exp_id}"
            if not os.path.isdir(os.path.join(DATA_DIR, exp_dir)):
                break
            exp_id += 1
        return exp_dir

    def save(self, obj, name):
        file_path = os.path.join(DATA_DIR, self.dir, f"{name}.pickle")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)

    def load(self, name):
        file_path = os.path.join(DATA_DIR, self.dir, f"{name}.pickle")
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def list(self, pattern='*'):
        file_path = os.path.join(DATA_DIR, self.dir, f"{pattern}.pickle")
        return [os.path.basename(path)[:-7] for path in glob.iglob(file_path)]
