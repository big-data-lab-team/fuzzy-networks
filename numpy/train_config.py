from layers import Convolution, Flatten, Dense, ReLU
import json

mlp_config = {
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
        'mnist_sample_shape': (784,)
    },
    'dataset_type': 'mnist',
    'eval_during_training': False,
}

cnn_config = {
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
        'mnist_sample_shape': (28, 28, 1)
    },
    'dataset_type': 'mnist',
    'eval_during_training': False,
}

string = json.dumps(cnn_config,)
print(string)