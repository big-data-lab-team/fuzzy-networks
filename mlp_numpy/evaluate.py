import utils
from neural_network import NN, load_mnist
import sys
import uuid

if len(sys.argv) < 2:
    print('The identifier of an experiment containing a trained neural network must be provided.')
    exit()

exp = utils.ExperimentResults(sys.argv[1])
nn = exp.load('neural_network')
train, val, test = load_mnist()

X, y = test
y_pred_proba = nn.predict_proba(X)

exp.save(y_pred_proba, f'test_predictions_{uuid.uuid4().hex}')
