import pickle
import os

DATA_DIR = 'results'

class ExperimentResults:
    def __init__(self, subdir=None):
        self.dir = subdir or self.get_experiment_dir()
        print(f'Saving results in {self.dir}')

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
