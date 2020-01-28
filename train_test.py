import os

possible_names = ['ResNet-18', 'ResNeXt-101-32x8d', 'WideResNet-101-2', 'WideResNet-50-2', 'ResNet-152',
                  'Densenet-161', 'ResNeXt-50-32x4d', 'ResNet-101', 'Densenet-201', 'ResNet-50',
                  'Densenet-169', 'Densenet-121', 'ResNet-34']

for name in possible_names:
    print('-' * 100)
    print('training {}'.format(name))
    os.system('python3 train_models.py --model_name {} --image_path {} --epochs'.format(name, '', 64))
    print('testing {}'.format(name))
    os.system('python3 test_models.py --image_path {} --model_name {}'.format('faces2/', name))
