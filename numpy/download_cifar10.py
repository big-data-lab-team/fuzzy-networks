import tensorflow as tf
import pickle
import os

data_file = 'cifar10.pickle'

if os.path.isfile(data_file):
    print('The cifar10 dataset is already downloaded.')
else:
    print('cifar10 is cannot be found locally, proceeding with installation...')
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    assert(len(train_images) == len(train_labels))
    with open(data_file, 'wb') as f:
        obj = train_images, train_labels, test_images, test_labels
        pickle.dump(obj, f)
    print('installation done')
