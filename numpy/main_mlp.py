import sys
import uuid
import numpy as np

import utils
from neural_network import NN, Dense, ReLU, load_mnist

def train():
    hyperparameters = {
        'architecture': (Dense(784), ReLU(), Dense(256), ReLU(), Dense(10)),
        'epsilon': 1e-6,
        'lr': 5e-2,
        'batch_size': 64,
        'n_epochs': 15
    }
    hyperparameters['seed'] = np.random.randint(1e5)

    mnist = load_mnist()
    nn = NN(data=mnist, **hyperparameters)

    train_logs = nn.train_loop(eval_acc=True)

    test_results = nn.evaluate()

    exp = utils.ExperimentResults()
    exp.save(train_logs, 'train_logs')
    exp.save(test_results, 'test_results')
    exp.save(hyperparameters, 'hyperparams')
    exp.save(nn, 'neural_network')


def evaluate(experiment_dir, docker_image_tag):
    exp = utils.ExperimentResults(experiment_dir)
    nn = exp.load('neural_network')
    train, val, test = load_mnist()

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
