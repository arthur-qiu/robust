import os
import glob
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from shutil import copyfile
from PIL import Image
symlink = False

cinic_directory = "/Users/qiuhaonan/Desktop/image_dataset/CINIC-10"
cifar_directory = "/Users/qiuhaonan/Desktop/image_dataset/cinic-10-cifar"
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
sets = ['train', 'valid', 'test']
if not os.path.exists(cifar_directory):
    os.makedirs(cifar_directory)
if not os.path.exists(cifar_directory + '/train'):
    os.makedirs(cifar_directory + '/train')
if not os.path.exists(cifar_directory + '/test'):
    os.makedirs(cifar_directory + '/test')

for c in classes:
    if not os.path.exists('{}/train/{}'.format(cifar_directory, c)):
        os.makedirs('{}/train/{}'.format(cifar_directory, c))
    if not os.path.exists('{}/test/{}'.format(cifar_directory, c)):
        os.makedirs('{}/test/{}'.format(cifar_directory, c))

location_id_mapping_train = {}
location_id_mapping_test = {}
for s in sets:
    for c in classes:
        source_directory = '{}/{}/{}'.format(cinic_directory, s, c)
        filenames = glob.glob('{}/cifar*.png'.format(source_directory))
        for fn in filenames:
            dest_fn = fn.split('/')[-1]
            if 'train' in fn:
                dest_fn = '{}/train/{}/{}'.format(cifar_directory, c, dest_fn)
                if symlink:
                    if not os.path.islink(dest_fn):
                        os.symlink(fn, dest_fn)
                else:
                    copyfile(fn, dest_fn)
                location_id_mapping_train[int(fn.split('/')[-1].split('.')[0].split('-')[-1])] = dest_fn

            elif 'test' in fn:
                dest_fn = '{}/test/{}/{}'.format(cifar_directory, c, dest_fn)
                if symlink:
                    if not os.path.islink(dest_fn):
                        os.symlink(fn, dest_fn)
                else:
                    copyfile(fn, dest_fn)

                location_id_mapping_test[int(fn.split('/')[-1].split('.')[0].split('-')[-1])] = dest_fn