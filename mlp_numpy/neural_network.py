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
                 architecture,
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=64,
                 seed=None,
                 activation="relu",
                 n_epochs=10,
                 data=None
                 ):

        self.architecture = architecture
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon
        self.default_n_epochs = n_epochs

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        self.train, self.valid, self.test = data


    def initialize_weights(self, dim_input):
        if self.seed is not None:
            np.random.seed(self.seed)
        dim_prev_layer_ouput = dim_input
        for layer in self.architecture:
            layer.initialize(dim_prev_layer_ouput)
            dim_prev_layer_ouput = layer.dim_ouput

    def softmax(self, x):
        x = x - x.max(axis=-1, keepdims=True) # To prevent overflow when applying exp to x
        exp_x = np.exp(x)
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

    def forward(self, x): 
        for layer in self.architecture:
            x = layer.forward(x)
        return self.softmax(x)

    def backward(self, outputs, labels):
        error = outputs - labels
        for layer in reversed(self.architecture):
            error = layer.backward(error)

    def update(self):
        for layer in self.architecture:
            layer.update(self.lr)

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
        return self.forward(X)

    def train_loop(self, n_epochs=None):
        n_epochs = n_epochs or self.default_n_epochs
        X_train, y_train = self.train
        y_onehot = y_train
        self.initialize_weights(X_train.shape[1:])

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                outputs = self.forward(minibatchX)
                self.backward(outputs, minibatchY)
                self.update()
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

class Dense:
    def __init__(self, n_neuron):
        self.n_neuron = n_neuron
        self.params = {}
        self.grads = {}
        self.cache = {}

    def forward(self, X):
        self.cache["X"] = X
        # import pdb ; pdb.set_trace()
        return X @ self.params["W"] + self.params["b"]

    def backward(self, dA):
        self.grads["dW"] = (self.cache['X'].T @ dA) / self.cache['X'].shape[0]
        self.grads["db"] = dA.mean(0, keepdims=True)
        return (self.params["W"] @ np.expand_dims(dA, 2)).squeeze()

    def initialize(self, dim_input):
        assert len(dim_input) == 1
        self.dim_ouput = (self.n_neuron,)
        bound = np.sqrt(6 / (dim_input[-1] + self.n_neuron))
        self.params["W"] = np.random.uniform(-bound, bound, size=(dim_input[-1], self.n_neuron))
        self.params["b"] = np.zeros((1, self.n_neuron))

    def update(self, lr):
        self.params["W"] -= lr * self.grads["dW"]
        self.params["b"] -= lr * self.grads["db"]


def conv2d(X, f):
    """
        X: input data (batch_n, H, W, channel)
        f: kernel weights (batch_n, H', W', channel_input, channel_output)    
    """
    shape_virtual_tensor = X.shape[:1] + f.shape[:2] + tuple(np.subtract(X.shape[1:3], f.shape[:2]) + 1) + X.shape[3:]
    strd = np.lib.stride_tricks.as_strided
    # import pdb; pdb.set_trace()
    virtual_tensor = strd(X, shape=shape_virtual_tensor, strides=X.strides[:1] + X.strides[1:3] * 2 + X.strides[3:])
    # print(virtual_tensor.shape)
    return np.einsum('ijmn,bijklm->bkln', f, virtual_tensor)
    # b: batch number
    # i: height in kernel
    # j: width in kernel
    # k: height of kernel in input
    # l: width of kernel in input
    # m: channel from input
    # n: channel from output    

class Convolution():
    def __init__(self, kernel_size, n_channel):
        self.kernel_size = tuple(kernel_size)
        self.n_channel = n_channel
        self.grads = {}
        self.cache = {}
        self.params = {}
    
    def forward(self, X):
        self.cache["X"] = X
        return conv2d(X, self.params["W"]) + self.params["b"]
        
    def initialize(self, dim_input):
        self.dim_output = tuple(np.substract(dim_input[:2], self.kernel_size) + 1) + (self.n_channel,)
        
        # Inspired from Keras https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/initializers.py#L462
        receptive_field_size = np.prod(self.kernel_size[0])
        fan_in = dim_input[-1] * receptive_field_size
        fan_out = self.n_channel * receptive_field_size
        bound = np.sqrt(6 / (fan_in + fan_out))
        self.params["W"] = np.random.uniform(-bound, bound, size=(self.kernel_size + (dim_input[-1],) + (self.n_channel,)))
        self.params["b"] = np.zeros((1, 1, 1, self.n_channel))
    
    def backward(self, error):
        # error dim: dim of output
        # See https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
        self.grads["dW"] = np.transpose(conv2d(np.transpose(self.cache["X"], [3, 1, 2, 0]), np.transpose(error, [1, 2, 0, 3])), [1, 2, 0, 3])
        self.grads["dB"] = error.sum(axis=(0, 1))

        return conv2d(error, self.params["W"].transpose(2, 3))


    def update(self, lr):
        self.params["W"] -= lr * self.grads["dW"]
        self.params["b"] -= lr * self.grads["db"]

class Pooling():
    pass


class ReLU:
    def __init__(self):
        self.grads = {}
        self.cache = {}
    def forward(self, A):
        self.cache["A"] = A 
        return np.maximum(0, A)
    def backward(self, dZ):
        return dZ * (self.cache["A"] > 0).astype(np.float64)
    def initialize(self, dim_input):
        self.dim_ouput = dim_input
    def update(self, lr):
        pass