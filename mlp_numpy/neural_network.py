import pickle
import numpy as np
import gzip

def one_hot(y, n_classes=10):
    return np.eye(n_classes)[y]

def load_mnist():
    data_file = gzip.open("mnist.pkl.gz", "rb")
    train_data, val_data, test_data = pickle.load(data_file, encoding="latin1")
    data_file.close()

    train_inputs = [np.reshape(x, (784, 1)) for x in train_data[0]]
    train_results = [one_hot(y, 10) for y in train_data[1]]
    train_data = np.array(train_inputs).reshape(-1, 784), np.array(train_results).reshape(-1, 10)

    val_inputs = [np.reshape(x, (784, 1)) for x in val_data[0]]
    val_results = [one_hot(y, 10) for y in val_data[1]]
    val_data = np.array(val_inputs).reshape(-1, 784), np.array(val_results).reshape(-1, 10)

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_results = [one_hot(y, 10) for y in test_data[1]]
    test_data = np.array(test_inputs).reshape(-1, 784), np.array(test_results).reshape(-1, 10)

    return train_data, val_data, test_data

class NN(object):
    def __init__(self,
                 hidden_dims=(784, 256), # 818970 parameters
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=64,
                 seed=None,
                 activation="relu",
                 init_method="glorot",
                 n_epochs=10,
                 data=None
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon
        self.init_method = init_method
        self.default_n_epochs = n_epochs

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        self.train, self.valid, self.test = data


    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            if self.init_method == 'glorot':
                bound = np.sqrt(6 / (all_dims[layer_n-1] + all_dims[layer_n]))
                self.weights[f"W{layer_n}"] = np.random.uniform(-bound, bound, size=all_dims[layer_n-1:layer_n+1])
            elif self.init_method == 'normal':
                self.weights[f"W{layer_n}"] = np.random.normal(0, 1, size=all_dims[layer_n-1:layer_n+1])
            elif self.init_method == 'zero':
                self.weights[f"W{layer_n}"] = np.zeros(all_dims[layer_n-1:layer_n+1])
            else:
                raise NotImplementedError(f'Activation method {self.init_method} not implemented')
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

    def relu(self, x, grad=False):
        if grad:
            return (x > 0).astype(np.float64)
        return np.maximum(0, x)

    def sigmoid(self, x, grad=False):
        sigma_x = 1 / (1 + np.exp(-x))
        if grad:
            return sigma_x * (1-sigma_x)
        return sigma_x

    def tanh(self, x, grad=False):
        tanh_x = np.tanh(x)
        if grad:
            return 1 - tanh_x**2
        return tanh_x

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            return self.relu(x, grad)
        elif self.activation_str == "sigmoid":
            return self.sigmoid(x, grad)
        elif self.activation_str == "tanh":
            return self.tanh(x, grad)
        else:
            raise Exception("invalid")
        return 0

    def softmax(self, x):
        x = x - x.max(axis=-1, keepdims=True) # To prevent overflow when applying exp to x
        exp_x = np.exp(x)
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

    def forward(self, x):
        cache = {"Z0": x}
        for layer_n in range(1, self.n_hidden + 1):
            cache[f"A{layer_n}"] = cache[f"Z{layer_n-1}"] @ self.weights[f"W{layer_n}"] + self.weights[f"b{layer_n}"]
            cache[f"Z{layer_n}"] = self.activation(cache[f"A{layer_n}"])
        layer_n = self.n_hidden + 1
        cache[f"A{layer_n}"] = cache[f"Z{layer_n-1}"] @ self.weights[f"W{layer_n}"] + self.weights[f"b{layer_n}"]
        cache[f"Z{layer_n}"] = self.softmax(cache[f"A{layer_n}"])

        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}
        grads[f"dA{self.n_hidden + 1}"] = output - labels
        for layer_n in range(self.n_hidden + 1, 1, -1):
            grads[f"dW{layer_n}"] = (cache[f"Z{layer_n-1}"].T @ grads[f"dA{layer_n}"]) / cache[f"Z{layer_n-1}"].shape[0]
            grads[f"db{layer_n}"] = grads[f"dA{layer_n}"].mean(0, keepdims=True)
            grads[f"dZ{layer_n-1}"] = (self.weights[f"W{layer_n}"] @ np.expand_dims(grads[f"dA{layer_n}"], 2)).squeeze()
            grads[f"dA{layer_n-1}"] = grads[f"dZ{layer_n-1}"] * self.activation(cache[f"A{layer_n-1}"], grad=True)
        grads[f"dW1"] = (cache[f"Z0"].T @ grads[f"dA1"]) / cache["Z0"].shape[0]
        grads[f"db1"] = grads[f"dA1"].mean(0, keepdims=True)

        return grads

    def update(self, grads):
        for layer_n in range(1, self.n_hidden + 2):
            self.weights[f"W{layer_n}"] -= self.lr * grads[f"dW{layer_n}"]
            self.weights[f"b{layer_n}"] -= self.lr * grads[f"db{layer_n}"]

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        return - (labels * np.log(prediction)).sum(axis=1).mean()

    def compute_loss_and_accuracy(self, X, y):
        one_y = y
        y = np.argmax(y, axis=1)  # Change y to integers
        proba = self.predict_proba(X)
        predictions = np.argmax(proba, axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(proba, one_y)

        return loss, accuracy, predictions

    def predict_proba(self, X):
        cache = self.forward(X)
        return cache[f"Z{self.n_hidden + 1}"]

    def train_loop(self, n_epochs=None):
        n_epochs = n_epochs or self.default_n_epochs
        X_train, y_train = self.train
        y_onehot = y_train
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                cache = self.forward(minibatchX)
                grads = self.backward(cache, minibatchY)
                self.update(grads)
                n_char = int(50*batch/n_batches)
                print('\r[' + '='*n_char + '>' + ' '*(49-n_char) + ']', end='')
            print('\r[' + '='*50 + ']')


            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)
            print(f"Epoch {epoch}: train_accuracy={train_accuracy:.3f}, valid_accuracy={valid_accuracy:.3f}, train_loss={train_loss:1.2e}, valid_loss={valid_loss:1.2e}")

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss, test_accuracy
