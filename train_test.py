import os
import time

from utils import get_config


def main():
    for name in get_config()['model_names']:
        print('-' * 100)
        
        print('Training {}'.format(name))
        start = time.perf_counter()
        os.system('python3 train_model.py {}'.format(name))
        print('Training {} complete'.format(name))
        print('Total time for training {:.5f} seconds'.format(time.perf_counter() - start))
        
        print('Testing {}'.format(name))
        start = time.perf_counter()
        os.system('python3 test_model.py {}'.format(name))
        print('Testing {} complete'.format(name))
        print('Total time for testing {:.5f} seconds'.format(time.perf_counter() - start))


if __name__ == '__main__':
    main()
