import sys
import uuid
import numpy as np

import utils
from neural_network import NN
from layers import Convolution, Flatten, Dense, ReLU
from data_prep import load_mnist, load_cifar10


def train(hyperparameters, sample_shape, nn_type):

    hyperparameters['seed'] = np.random.randint(1e5)

    cifar10 = load_cifar10(flatten_input=(nn_type == 'mlp'))  # because mlp only processes 1D input
    nn = NN(data=cifar10, **hyperparameters)

    perform_evaluation = False
    if nn_type == 'mlp':
        perform_evaluation = True
    elif hyperparameters['n_epochs'] == 1:
        perform_evaluation = True

    train_logs = nn.train_loop(eval_each_epoch=perform_evaluation)

    exp = utils.ExperimentResults()
    exp.save(train_logs, 'train_logs')
    exp.save(hyperparameters, 'hyperparams')
    exp.save(nn, 'neural_network')

    test_results = nn.evaluate()
    exp.save(test_results, 'test_results')


def evaluate(experiment_dir, docker_image_tag, sample_shape):
    exp = utils.ExperimentResults(experiment_dir)
    nn = exp.load('neural_network')
    train, val, test = nn.train, nn.valid, nn.test

    X, y = test
    X = X[:100]  # Using all test examples would take too long
    y_pred_proba = nn.predict_proba(X)

    result_name = f'test_predictions_{docker_image_tag}_{uuid.uuid4().hex}'
    exp.save(y_pred_proba, result_name)


if len(sys.argv) < 3:
    print('Specify the type of neural network: cnn, mlp and whether to train or evaluate')
neural_network_type = sys.argv[1]

if neural_network_type == 'mlp':
    hyperparameters = {
        'architecture': (Dense(1024), ReLU(), Dense(256), ReLU(), Dense(10)),  # 1024 because 32x32 for cifar10
        'epsilon': 1e-6,
        'lr': 5e-2,
        'batch_size': 64,
        'n_epochs': 100
    }
    sample_shape = (784,)
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

    hyperparameters = {
        'architecture': architecture,
        'epsilon': 1e-6,
        'lr': 5e-2,
        'batch_size': 64,
        'n_epochs': 1
    }
    sample_shape = (28, 28, 1)
else:
    raise ValueError('Unknown neural network type')

if sys.argv[2] == 'evaluate':
    if len(sys.argv) < 3:
        print('The identifier of an experiment containing a trained neural'
              ' network must be provided.')
        exit()
    if len(sys.argv) < 4:
        docker_image_tag = 'no-mca'
    else:
        docker_image_tag = sys.argv[4].replace(' ', '_')
    experiment_dir = sys.argv[3]
    evaluate(experiment_dir, docker_image_tag, sample_shape)
elif sys.argv[2] == 'train':
    train(hyperparameters, sample_shape, neural_network_type)
else:
    raise ValueError(f'Unknown operation {sys.argv[2]}')
