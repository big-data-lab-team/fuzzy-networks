import pickle
import os
import pprint

# pp = pprint.PrettyPrinter(indent=4)


def convert_time(seconds):
    '''This function takes a time in seconds and converts it
    to the hours, minutes, seconds format'''
    return f'00:00:{int(round(seconds,0))}'


def generate_csv(results_dir='./results'):
    report_file = 'benchmark_report.csv'
    dirs = os.listdir(results_dir)
    csv_str = 'type of network, dataset, learning rate, batch size, epochs, training time (hh:mm:ss), accuracy (%)\n'

    if os.path.isfile(report_file):
        answer = input(f'The {report_file} already exists, do you wish to overwrite it? (y/n)')
        if answer.lower() not in ['y', 'yes']:
            print('File will not be overwritten, exiting now.')
            return

    for experiment in dirs:

        config_file = f'{results_dir}/{experiment}/config.pickle'
        accuracy_file = f'{results_dir}/{experiment}/test_results.pickle'
        runtime_file = f'{results_dir}/{experiment}/training_runtime.pickle'

        exists = os.path.isfile

        if exists(config_file) and exists(accuracy_file) and exists(runtime_file):
            with open(config_file, 'rb') as f:
                config = pickle.load(f)
                new_line = f"{config['nn_type']},"\
                    f"{config['dataset_name']},"\
                    f"{config['hyperparameters']['lr']},"\
                    f"{config['hyperparameters']['batch_size']}, "\
                    f"{config['hyperparameters']['n_epochs']},"

            with open(runtime_file, 'rb') as f:
                runtime = convert_time(pickle.load(f))
                new_line += f'{runtime},'

            with open(accuracy_file, 'rb') as f:
                accuracy = pickle.load(f)[1]
                accuracy = round(accuracy*100, 4)
                new_line += f'{accuracy},\n'

            csv_str += new_line

            with open(report_file, 'w') as f:
                f.write(csv_str)
            
            print(f'Generated report can be found in {report_file}')


if __name__ == '__main__':
    generate_csv()
