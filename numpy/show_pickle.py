import sys
import pickle

from neural_network import NN
from layers import Convolution, Flatten, Dense, ReLU

file_to_unpickle = sys.argv[1]

with open(file_to_unpickle, 'rb') as target:
    obj = pickle.load(target)
    print(obj)
