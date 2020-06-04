import numpy as np
from cifar10_prep import load_cifar10
import utils
from neural_network import NN, Convolution, Flatten, Dense, ReLU, load_mnist

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
hyperparameters['seed'] = np.random.randint(1e5)

mnist = load_mnist(shape=(28, 28, 1))
nn = NN(data=mnist, **hyperparameters)

train_logs = nn.train_loop()


exp = utils.ExperimentResults()
exp.save(hyperparameters, 'hyperparams')
exp.save(train_logs, 'train_logs')
exp.save(nn, 'neural_network')
test_results = nn.evaluate()
exp.save(test_results, 'test_results')
print(test_results)
