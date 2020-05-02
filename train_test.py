import os
import time

from utils import (
    get_config,
    print_log,
)


def main():
    for name in get_config()['model_names']:
        print_log('-' * 100)
        
        print_log('Training {}'.format(name))
        start = time.perf_counter()
        os.system('python3 train_model.py {}'.format(name))
        print_log('Training {} complete'.format(name))
        print_log('Total time for training {:.5f} seconds'.format(time.perf_counter() - start))
        
        print_log('Testing {}'.format(name))
        start = time.perf_counter()
        os.system('python3 test_model.py {}'.format(name))
        print_log('Testing {} complete'.format(name))
        print_log('Total time for testing {:.5f} seconds'.format(time.perf_counter() - start))


if __name__ == '__main__':
    main()
