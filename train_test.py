import argparse
import os
import time

from utils import model_names


def get_argv():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--epochs_train', type=int)
    
    argv = parser.parse_args()
    return argv


def main():
    argv = get_argv()
    possible_names = model_names()
    
    for name in possible_names:
        print('-' * 100)
        
        print('Training {}'.format(name))
        start = time.perf_counter()
        os.system(
            'python3 train_model.py --model_name {} --image_path {} --epochs {}'.format(
                name,
                argv.train_path,
                argv.epochs_train
            )
        )
        print('Training {} complete')
        print('Total time for training {:.5f} seconds'.format(time.perf_counter() - start))
        
        with open('models_data/{}_BATCH.txt'.format(name), 'r') as f:
            batch_size = int(f.read())
        os.remove('models_data/{}_BATCH.txt'.format(name))
        
        print('t'
              'Testing {}'.format(name))
        start = time.perf_counter()
        os.system(
            'python3 test_model.py --model_name {} --image_path {} --batch_size'.format(
                argv.test_path,
                name,
                batch_size
            )
        )
        print('Testing {} complete'.format(name))
        print('Total time for testing {:.5f} seconds'.format(time.perf_counter() - start))


if __name__ == '__main__':
    main()
