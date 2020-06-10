import pickle
import gzip
import numpy as np
import tensorflow as tf
from collections import defaultdict


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


def load_cifar10(flatten_input=False):
    '''For MNIST the input imagines need to flattened whereas for CNN they have to be multi-dimensional'''

    def flatten(grayscaled):
        # Originally 32x32x1 grayscaled numpy array image
        return np.array([np.reshape(x, (32*32,)) for x in grayscaled])

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    assert(len(train_images) == len(train_labels))

    train_images = tf.image.rgb_to_grayscale(train_images).numpy()
    test_images = tf.image.rgb_to_grayscale(test_images).numpy()

    if flatten_input:
        train_images = flatten(train_images)
        test_images = flatten(test_images)

    size = len(train_images)
    # randomly choose 20% of indices to be taken out of training set
    validation_indices = sorted(np.random.choice(size, int(size*0.2), replace=False))

    indices = defaultdict(int)
    for index in validation_indices:
        indices[index] = True

    def is_validation_index(i, dic): return dic[i]

    new_train_images = np.array([train_images[i] for i in range(size) if not is_validation_index(i, indices)])
    new_train_labels = np.array([train_labels[i] for i in range(size) if not is_validation_index(i, indices)])

    assert(len(new_train_images) == size*0.8)
    assert(len(new_train_labels) == size*0.8)

    valid_images = np.array([train_images[i] for i in range(size) if is_validation_index(i, indices)])
    valid_labels = np.array([train_labels[i] for i in range(size) if is_validation_index(i, indices)])

    assert(len(valid_images) == size*0.2)
    assert(len(valid_labels) == size*0.2)

    def normalize(array):
        return array.astype('float32') / 255.0

    def one_hot(array, n_classes=10):
        return np.eye(n_classes)[array].reshape(len(array), n_classes)

    training = normalize(new_train_images), one_hot(new_train_labels)
    validation = normalize(valid_images), one_hot(valid_labels)
    testing = normalize(test_images), one_hot(test_labels)

    return training, validation, testing
