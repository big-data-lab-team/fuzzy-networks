import pickle
import gzip
import numpy as np
import tensorflow as tf



def load_mnist(shape=(784,)):
    
    def one_hot(y, n_classes=10):
        return np.eye(n_classes)[y]
    
    data_file = gzip.open("mnist.pkl.gz", "rb")
    train_data, val_data, test_data = pickle.load(data_file, encoding="latin1")
    data_file.close()

    train_inputs = [np.reshape(x, (784, 1)) for x in train_data[0]]
    train_results = [one_hot(y, 10) for y in train_data[1]]
    train_data = (np.array(train_inputs).reshape(-1, *shape),
                  np.array(train_results).reshape(-1, 10))

    val_inputs = [np.reshape(x, (784, 1)) for x in val_data[0]]
    val_results = [one_hot(y, 10) for y in val_data[1]]
    val_data = (np.array(val_inputs).reshape(-1, *shape),
                np.array(val_results).reshape(-1, 10))

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_results = [one_hot(y, 10) for y in test_data[1]]
    test_data = (np.array(test_inputs).reshape(-1, *shape),
                 np.array(test_results).reshape(-1, 10))

    return train_data, val_data, test_data

