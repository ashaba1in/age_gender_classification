import os
import time

possible_names = ['ResNet-18', 'Densenet-121', 'Densenet-201']

for name in possible_names:
    print('-' * 100)
    print('training {}'.format(name))
    start = time.perf_counter()
    os.system('python3 train_model.py --model_name {} --image_path {} --epochs {}'.format(name, 'aligned/', 72))
    print('Total time for training {:.5f} seconds'.format(time.perf_counter() - start))
    print('testing {}'.format(name))
    os.system('python3 test_model.py --image_path {} --model_name {}'.format('faces2/', name))
