# Numerical stability of a simple MLP on MNIST dataset

This directory contains an implementation of a simple Multilayer perceptron with Numpy.
The use of NumPy makes it easy to perform inference using Monte Carlo arithmetic using [Fuzzy docker images](https://github.com/gkiar/fuzzy "Fuzzy Github project").

## How to use ?
 1. Train the simple fully-connected neural network with `python main.py`
 2. Use the Jupyter notebook `evaluate_stability.ipynb` to evaluate the number of significant digits of the probabilities predicted by the MLP for the first 100 test examples.
    Specify at the beginning of the notebook: the name of the experiment where the trained neural network has been saved, the number of trials to run and the components of the stack using MCA :
    ```python
    EXP_DIR = 'exp_2'        # Directory of the experiment containing the trained neural network to use 
    N_TRIALS = 20            # Number of MCA trials to perform
    MCA_TAG = 'python-numpy' # Tag of the fuzzy docker image to use (python, python-numpy, etc.). It corresponds to the use of MCA in different parts of the stack
     ```
     
## Content
 This directory contains the following files:
 - `evaluate_stability.ipynb`: a Jupyter notebook to evaluate the number of significant digits of the probabilities predicted by the MLP
 - `evaluate.py`: a Python script to predict probabilities for the 100 first examples of the test set of MNIST
 - `main.py`: a Python script to train a simple MLP on MNIST
 - `mnist.pkl.gz`: MNIST dataset
 - `neural_network.py`: implementation of a simple MLP with NumPy
 - `predict_with_mca.sh`: shell script to run evaluate.py inside a Fuzzy docker image
 - `utils.py`: Python module to save and load results from experiments
 