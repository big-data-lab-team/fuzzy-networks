import numpy as np


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

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [],
                           'train_loss': [], 'validation_loss': []}

        self.train, self.valid, self.test = data

    def initialize_weights(self, dim_input):
        if self.seed is not None:
            np.random.seed(self.seed)
        dim_prev_layer_output = dim_input
        for layer in self.architecture:
            layer.initialize(dim_prev_layer_output)
            dim_prev_layer_output = layer.dim_output

    def softmax(self, x):
        # Preprocess x to prevent overflow when applying exp to x
        x = x - x.max(axis=-1, keepdims=True)
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

    def train_loop(self, n_epochs=None, eval_each_epoch=False):
        n_epochs = n_epochs or self.default_n_epochs
        X_train, y_train = self.train
        y_onehot = y_train
        self.initialize_weights(X_train.shape[1:])

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                minibatch_slice = \
                    slice(self.batch_size * batch, self.batch_size * (batch + 1))
                minibatchX = X_train[minibatch_slice]
                minibatchY = y_onehot[minibatch_slice]
                outputs = self.forward(minibatchX)
                self.backward(outputs, minibatchY)
                self.update()
                n_char = int(50 * batch / n_batches)
                print('\r[' + '=' * n_char + '>' + ' ' * (49 - n_char) + ']',
                      end='')
            print('\rTraining: [' + '=' * 50 + ']')

            if eval_each_epoch:
                self.log_performances(epoch)
        if not eval_each_epoch:
            self.log_performances(epoch)

        return self.train_logs

    def log_performances(self, epoch):
        X_train, y_train = self.train
        train_loss, train_accuracy = self.compute_loss_and_accuracy(X_train, y_train)
        X_valid, y_valid = self.valid
        valid_loss, valid_accuracy = self.compute_loss_and_accuracy(X_valid, y_valid)

        self.train_logs['train_accuracy'].append(train_accuracy)
        self.train_logs['validation_accuracy'].append(valid_accuracy)
        self.train_logs['train_loss'].append(train_loss)
        self.train_logs['validation_loss'].append(valid_loss)
        print(f"Epoch {epoch}: train_accuracy={train_accuracy:.3f}, "
              f"valid_accuracy={valid_accuracy:.3f}, "
              f"train_loss={train_loss:1.2e},"
              f"valid_loss={valid_loss:1.2e}")

    def evaluate(self):
        X_test, y_test = self.test
        test_loss, test_accuracy, _ = \
            self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss, test_accuracy
