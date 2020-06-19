import numpy as np
import einsum2

class Dense:
    def __init__(self, n_neuron):
        self.n_neuron = n_neuron
        self.params = {}
        self.grads = {}
        self.cache = {}

    def forward(self, X):
        self.cache["X"] = X
        return X @ self.params["W"] + self.params["b"]

    def backward(self, dA):
        self.grads["dW"] = (self.cache['X'].T @ dA) / self.cache['X'].shape[0]
        self.grads["db"] = dA.mean(0, keepdims=True)
        return (self.params["W"] @ np.expand_dims(dA, 2)).squeeze()

    def initialize(self, dim_input):
        assert len(dim_input) == 1
        self.dim_output = (self.n_neuron,)
        bound = np.sqrt(6 / (dim_input[-1] + self.n_neuron))
        self.params["W"] = \
            np.random.uniform(-bound, bound, size=(dim_input[-1], self.n_neuron))
        self.params["b"] = np.zeros((1, self.n_neuron))

    def update(self, lr):
        self.params["W"] -= lr * self.grads["dW"]
        self.params["b"] -= lr * self.grads["db"]


def conv2d(X, f):
    """
        X: input data (batch_n, H, W, channel)
        f: kernel weights (batch_n, H', W', channel_input, channel_output)
    """
    shape_virtual_tensor = (X.shape[:1]
                            + f.shape[:2]
                            + tuple(np.subtract(X.shape[1:3], f.shape[:2]) + 1)
                            + X.shape[3:])
    strides_virtual_tensor = X.strides[:1] + X.strides[1:3] * 2 + X.strides[3:]
    strd = np.lib.stride_tricks.as_strided
    virtual_tensor = strd(X, shape=shape_virtual_tensor,
                          strides=strides_virtual_tensor)
    return einsum2.einsum2('ijmn,bijklm->bkln', f, virtual_tensor)
    # b: batch number
    # i: height in kernel
    # j: width in kernel
    # k: height of kernel in input
    # l: width of kernel in input
    # m: channel from input
    # n: channel from output


class Convolution():
    def __init__(self, kernel_size, n_channel):
        if type(kernel_size) == int:
            kernel_size = (kernel_size, kernel_size)
        assert (kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1)
        self.kernel_size = tuple(kernel_size)
        self.n_channel = n_channel
        self.grads = {}
        self.cache = {}
        self.params = {}

    def forward(self, X):
        self.cache["X"] = X
        return conv2d(X, self.params["W"]) + self.params["b"]

    def initialize(self, dim_input):
        self.dim_output = \
            tuple(np.subtract(dim_input[:2], self.kernel_size) + 1) \
            + (self.n_channel,)
        # Inspired from the way Keras intialize by default:
        receptive_field_size = np.prod(self.kernel_size[0])
        fan_in = dim_input[-1] * receptive_field_size
        fan_out = self.n_channel * receptive_field_size
        bound = np.sqrt(6 / (fan_in + fan_out))
        weight_shape = (self.kernel_size + (dim_input[-1],) + (self.n_channel,))
        self.params["W"] = np.random.uniform(-bound, bound, size=weight_shape)
        self.params["b"] = np.zeros((1, 1, 1, self.n_channel))

    def backward(self, error):
        # https://www.jefkine.com/general/2016/09/05/
        # backpropagation-in-convolutional-neural-networks/
        self.grads["dW"] = np.transpose(conv2d(
            np.transpose(self.cache["X"], [3, 1, 2, 0]),
            np.transpose(error, [1, 2, 0, 3])), [1, 2, 0, 3])
        self.grads["dW"] /= error.shape[0]
        # Sum across image height and width and average across the batch
        self.grads["db"] = error.sum(axis=(1, 2)).mean(axis=0)

        # No padding for first and last dimension
        padding = (0, *np.subtract(self.kernel_size, 1), 0)
        # np.pad expect two numbers for each dimension (left, right)
        padding = np.tile(np.array(padding)[:, None], 2)
        return conv2d(np.pad(error, padding), np.transpose(self.params["W"], [0, 1, 3, 2]))

    def update(self, lr):
        self.params["W"] -= lr * self.grads["dW"]
        self.params["b"] -= lr * self.grads["db"]


class Flatten():
    def __init__(self):
        self.cache = {}

    def forward(self, X):
        return X.reshape(X.shape[0], *self.dim_output)

    def backward(self, error):
        return error.reshape(error.shape[0], *self.dim_input)

    def update(self, lr):
        pass

    def initialize(self, dim_input):
        self.dim_input = dim_input
        self.dim_output = (np.prod(dim_input),)


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
        self.dim_output = dim_input

    def update(self, lr):
        pass
