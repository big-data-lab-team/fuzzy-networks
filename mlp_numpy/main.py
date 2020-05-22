import pickle
import os
import numpy as np

import utils
from neural_network import NN, load_mnist

hyperparameters = {
    'hidden_dims': (784, 256),
    'epsilon': 1e-6,
    'lr': 5e-2,
    'batch_size': 64,
    'activation': "relu",
    'init_method': "glorot",
    'n_epochs': 15
}
hyperparameters['seed'] = np.random.randint(1e5)

mnist = load_mnist()
nn = NN(data=mnist, **hyperparameters)

train_logs = nn.train_loop()

test_results = nn.evaluate()

exp = utils.ExperimentResults()
exp.save(train_logs, 'train_logs')
exp.save(test_results, 'test_results')
exp.save(hyperparameters, 'hyperparams')
exp.save(nn, 'neural_network')
