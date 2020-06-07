# Numerical stability of a simple MLP on MNIST dataset

This directory contains an implementation of a simple Multilayer perceptron with Numpy.
The use of NumPy makes it easy to perform inference using Monte Carlo arithmetic using [Fuzzy docker images](https://github.com/gkiar/fuzzy "Fuzzy Github project").

## How to use ?
 1. Train the simple fully-connected neural network with `python main.py`
 2. Use the Jupyter notebook `evaluate_stability.ipynb` to evaluate the number of significant digits of the probabilities predicted by the MLP or CNN for the first 100 test examples.
    Specify at the beginning of the notebook: the name of the experiment where the trained neural network has been saved, the number of trials to run and the components of the stack using MCA :
    ```python
    EXP_DIR = 'exp_2'        # Directory of the experiment containing the trained neural network to use 
    N_TRIALS = 20            # Number of MCA trials to perform
    MCA_TAG = 'python-numpy' # Tag of the fuzzy docker image to use (python, python-numpy, etc.). It corresponds to the use of MCA in different parts of the stack
    NN_TYPE = "cnn"          # Type of neural network: mlp or cnn 
    PRECISION_64 = 53        # Default: 53
    PRECISION_32 = 24        # Default: 24
      ```
     
## Content
 This directory contains the following files:
 - `evaluate_stability.ipynb`: a Jupyter notebook to evaluate the number of significant digits of the probabilities predicted by the MLP
 - `mnist.pkl.gz`: MNIST dataset
 - `neural_network.py`: implementation of a simple MLP with NumPy
 - `predict_with_mca.sh`: shell script to run MCA simulations inside a Fuzzy docker image
 - `utils.py`: Python module to save and load results from experiments
 - `main.py`: a Python script to train and evaluate a simple MLP or CNN on MNIST 

you can use the main.py script to train a network by running
  ```
 $ python main.py [mlp|cnn] train
   ```
 you can use the main.py script to evaluate the an experiment by running
 ```
 $ python main.py [mlp|cnn] evaluate [experiment folder] "[precision 64]_[precision32]"
 ```
 for example `python main.py mlp evaluate exp_1 "53_24"`
 