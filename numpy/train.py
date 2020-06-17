import sys
import numpy as np

import utils
from data_prep import load_mnist, load_cifar10
from neural_network import NN
from layers import Convolution, Flatten, Dense, ReLU

err_msg = 'First argument needs to be network type: "mlp" or "cnn". ' +\
    'Second argument needs to be the dataset to train on: "mnist" or "cifar10"'


def train(hyperparameters,
          nn_type,
          dataset_name,
          sample_shape,
          flatten_cifar10,
          eval_while_training=True):

    hyperparameters['seed'] = np.random.randint(1e5)

    if dataset_name == 'mnist':
        dataset = load_mnist(sample_shape)

    elif dataset_name == 'cifar10':
        dataset = load_cifar10(flatten_input=flatten_cifar10)

    else:
        raise ValueError(err_msg)

    nn = NN(data=dataset, **hyperparameters)

    train_logs = nn.train_loop(eval_each_epoch=eval_while_training)

    exp = utils.ExperimentResults()
    exp.save(train_logs, 'train_logs')
    exp.save(hyperparameters, 'hyperparams')
    exp.save(nn, 'neural_network')

    test_results = nn.evaluate()
    exp.save(test_results, 'test_results')


try:
    neural_network_type = sys.argv[1]
    dataset_type = sys.argv[2]
except:
    raise Exception(err_msg)

if neural_network_type == 'mlp':
    hyperparameters = {
        'architecture': (
            Dense(1024),
            ReLU(),
            Dense(256),
            ReLU(),
            Dense(10)),  # 1024 because 32x32 for cifar10
        'epsilon': 1e-6,
        'lr': 5e-2,
        'batch_size': 64,
        'n_epochs': 3
    }

elif neural_network_type == 'cnn':
    architecture = (
        Convolution(3, 32),
        ReLU(),
        Convolution(3, 32),
        ReLU(),
        Convolution(3, 32),
        ReLU(),
        Flatten(),
        Dense(10)
    )
    mnist_sample_shape = (784,)

    hyperparameters = {
        'architecture': architecture,
        'epsilon': 1e-6,
        'lr': 5e-2,
        'batch_size': 64,
        'n_epochs': 1
    }
    mnist_sample_shape = (28, 28, 1)

else:
    raise ValueError(err_msg)

train(hyperparameters,
      neural_network_type,
      dataset_type,
      sample_shape=mnist_sample_shape,
      flatten_cifar10=(neural_network_type == 'mlp'))
