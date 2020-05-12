import os
import time

from utils import (
    get_config,
    print_log,
)


def main():
    for name in get_config()['model_names']:
        print_log('-' * 100)

        print_log('Processing model {}'.format(name))
        start = time.perf_counter()
        os.system('python3 process_model.py {}'.format(name))
        print_log('Processing model {} complete'.format(name))
        print_log('Total time {:.5f} seconds'.format(time.perf_counter() - start))


if __name__ == '__main__':
    main()
