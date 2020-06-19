import train
from train_config import mlp_config, cnn_config

epoch_run = [1, 10, 50, 100, 500,1000]

for run in epoch_run:
    mlp_config['hyperparameters']['n_epochs'] = run
    train.train(mlp_config)
    print(f'\n\n\n*********Done training for {run} epochs*********\n\n\n')
