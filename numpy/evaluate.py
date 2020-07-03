import sys
import uuid
import numpy as np

import utils
from data_prep import load_mnist, load_cifar10


err_msg = 'First argument should be the experiment folder you wish to evaluate.\n'\
    'Second argument should be the docker image tag used to perform the evaluation.\n'\
    'Last argument (optional) is the size of the subsample you wish to use. Default value takes the whole test set'


def evaluate(experiment_dir, docker_image_tag, subsample_size):
    exp = utils.ExperimentResults(experiment_dir)
    nn = exp.load('neural_network')
    test = nn.test

    X, y = test

    if subsample_size != None:
        X = X[:subsample_size]  # Using all test examples would take too long
    y_pred_proba = nn.predict_proba(X)

    result_name = f'test_predictions_{docker_image_tag}_{uuid.uuid4().hex}'
    exp.save(y_pred_proba, result_name)

try:
    experiment_folder = sys.argv[1]
    docker_tag = sys.argv[2]
except:
    raise ValueError(err_msg)

try:
    subsample = int(sys.argv[3])
except:
    subsample = None

evaluate(experiment_folder, docker_tag, subsample)
