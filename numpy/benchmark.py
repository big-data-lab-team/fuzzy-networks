import train
from train_config import mlp_config, cnn_config
import copy

epoch_run = [1, 10, 50, 100, 200]

mlp_mnist = copy.deepcopy(mlp_config)
mlp_cifar10 = copy.deepcopy(mlp_config)
cnn_mnist = copy.deepcopy(cnn_config)
cnn_cifar10 = copy.deepcopy(cnn_config)


for run in epoch_run:
    mlp_mnist['hyperparameters']['n_epochs'] = run
    mlp_cifar10['hyperparameters']['n_epochs'] = run
    mlp_cifar10['dataset_name'] = 'cifar10'
    
    cnn_mnist['hyperparameters']['n_epochs'] = run
    cnn_cnn['hyperparameters']['n_epochs'] = run
    cnn_cifar10['dataset_name'] = 'cifar10'
    
    train.train(mlp_mnist)
    print(f'\n\n\n*********Done training mlp on mnist for {run} epochs*********\n\n\n')
    
    train.train(mlp_cifar10)
    print(f'\n\n\n*********Done training mlp on cifar10 for {run} epochs*********\n\n\n')

    train.train(cnn_mnist)
    print(f'\n\n\n*********Done training cnn on mnist for {run} epochs*********\n\n\n')

    train.train(cnn_cifar10)
    print(f'\n\n\n*********Done training cnn on cifar10 for {run} epochs*********\n\n\n')


print("\n\n\n***----***All training is done")