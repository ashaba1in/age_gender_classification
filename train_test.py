import os
import time

from utils import model_names

possible_names = model_names()

for name in possible_names:
    print('-' * 100)
    
    start = time.perf_counter()
    print('training {}'.format(name))
    os.system('python3 train_model.py --model_name {} --image_path {} --epochs {}'.format(name, 'aligned/', 72))
    print('Total time for training {:.5f} seconds'.format(time.perf_counter() - start))
    
    with open('models_data/{}_BATCH.txt'.format(name), 'r') as f:
        batch_size = int(f.read())
    
    print('testing {}'.format(name))
    os.system('python3 test_model.py --model_name {} --image_path {} --batch_size'.format('faces2/', name, batch_size))
    print('testing {} complete'.format(name))
