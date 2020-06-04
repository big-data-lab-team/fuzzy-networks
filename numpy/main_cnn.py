import sys
import numpy as np
import utils
from neural_network import NN, Convolution, Flatten, Dense, ReLU, load_mnist

mnist = load_mnist(shape=(28, 28, 1))


def train():
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

    nn = NN(data=mnist, **hyperparameters)

    train_logs = nn.train_loop()

    exp = utils.ExperimentResults()
    exp.save(hyperparameters, 'hyperparams')
    exp.save(train_logs, 'train_logs')
    exp.save(nn, 'neural_network')
    test_results = nn.evaluate()
    exp.save(test_results, 'test_results')
    print(test_results)


def evaluate(experiment_dir, docker_image_tag):
    exp = utils.ExperimentResults(experiment_dir)
    nn = exp.load('neural_network')
    train, val, test = mnist

    X, y = test
    X = X[:100]  # Using all test examples would take too long
    y_pred_proba = nn.predict_proba(X)

    result_name = f'test_predictions_{docker_image_tag}_{uuid.uuid4().hex}'
    exp.save(y_pred_proba, result_name)


if len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
    if len(sys.argv) < 3:
        print('The identifier of an experiment containing a trained neural'
              ' network must be provided.')
        exit()
    if len(sys.argv) < 4:
        docker_image_tag = 'no-mca'
    else:
        docker_image_tag = sys.argv[3].replace(' ', '_')
    experiment_dir = sys.argv[2]
    evaluate(experiment_dir, docker_image_tag)
else:
    train()

