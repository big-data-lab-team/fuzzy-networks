from layers import Convolution, Flatten, Dense, ReLU
import json
import pprint


mlp_config = {
    'nn_type': 'mlp',
    'dataset_name': 'mnist',
    'eval_while_training': False,
    'mnist_sample_shape': (784,),
    'hyperparameters': {
        'architecture': (
            Dense(1024),
            ReLU(),
            Dense(256),
            ReLU(),
            Dense(10)  # 1024 because 32x32 for cifar10
        ),
        'epsilon': 1e-6,
        'lr': 5e-2,
        'batch_size': 64,
        'n_epochs': 3,
    },
}

cnn_config = {
    'nn_type': 'cnn',
    'dataset_name': 'mnist',
    'eval_while_training': False,
    'mnist_sample_shape': (28, 28, 1),
    'hyperparameters': {
        'architecture': (
            Convolution(3, 32),
            ReLU(),
            Convolution(3, 32),
            ReLU(),
            Convolution(3, 32),
            ReLU(),
            Flatten(),
            Dense(10)
        ),
        'epsilon': 1e-6,
        'lr': 5e-2,
        'batch_size': 64,
        'n_epochs': 1,
    },
}
