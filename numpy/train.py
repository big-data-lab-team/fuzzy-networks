import sys
import numpy as np
import timeit

import utils
from data_prep import load_mnist, load_cifar10
from neural_network import NN
from layers import Convolution, Flatten, Dense, ReLU
from train_config import mlp_config, cnn_config

err_msg = 'First argument needs to be network type: "mlp" or "cnn". ' +\
    'Second argument (optional) needs to be the dataset to train on: "mnist" or "cifar10"'


def train(config):
    '''The config object required for training is in train_config.py'''

    config['hyperparameters']['seed'] = np.random.randint(1e5)

    if config['dataset_name'] == 'mnist':
        dataset = load_mnist(config['mnist_sample_shape'])

    elif config['dataset_name'] == 'cifar10':
        dataset = load_cifar10(flatten_input=(config['nn_type'] == 'mlp'))

    else:
        raise ValueError(err_msg)

    nn = NN(data=dataset, **config['hyperparameters'])

    start = timeit.default_timer()
    
    train_logs = nn.train_loop(eval_each_epoch=config['eval_while_training'])
    
    elapsed = round(timeit.default_timer()-start,4)
    print(f'training runtime: {elapsed} seconds')

    exp = utils.ExperimentResults()
    exp.save(train_logs, 'train_logs')
    exp.save(config, 'config')
    exp.save(nn, 'neural_network')
    exp.save(elapsed,'training_runtime')

    test_results = nn.evaluate()
    exp.save(test_results, 'test_results')


try:
    neural_network_type = sys.argv[1]  # necessary to specify the nn type
except:
    raise Exception(err_msg)

if neural_network_type == 'mlp':
    config = mlp_config
elif neural_network_type == 'cnn':
    config = cnn_config  # defined in train_config
else:
    raise ValueError(err_msg)

try:
    dataset_name = sys.argv[2]
    assert(dataset_name == 'mnist' or dataset_name == 'cifar10')
    config['dataset_name'] = dataset_name
except AssertionError as e:
    raise AssertionError(err_msg)
except:
    pass  # if no dataset_name provided, the default is taken


train(config)
