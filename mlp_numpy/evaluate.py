import utils
from neural_network import NN, load_mnist
import sys
import uuid

if len(sys.argv) < 2:
    print('The identifier of an experiment containing a trained neural network must be provided.')
    exit()
if len(sys.argv) < 3:
    docker_image_tag = 'no-mca'
else:
    docker_image_tag = sys.argv[2].replace(' ', '_')
experiment_dir = sys.argv[1]


exp = utils.ExperimentResults(experiment_dir)
nn = exp.load('neural_network')
train, val, test = load_mnist()

X, y = test
X = X[:100] # Using all test examples would take too long
y_pred_proba = nn.predict_proba(X)

result_name = f'test_predictions_{docker_image_tag}_{uuid.uuid4().hex}'
exp.save(y_pred_proba, result_name)
