import os
import glob
import numpy as np
from shutil import copyfile
symlink = False    # If this is false the files are copied instead
combine_train_valid = False    # If this is true, the train and valid sets are ALSO combined
combine_all = True
import random
random.seed(1)

cinic_directory = "/Users/qiuhaonan/Desktop/image_dataset/cinic-10-cifar"
imagenet_directory = "/Users/qiuhaonan/Desktop/image_dataset/cinic-10-cifar"
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
sets = ['train']
if not os.path.exists(imagenet_directory):
    os.makedirs(imagenet_directory)
if not os.path.exists(imagenet_directory + '/train10k3'):
    os.makedirs(imagenet_directory + '/train10k3')

if not os.path.exists(imagenet_directory + '/train10k4'):
    os.makedirs(imagenet_directory + '/train10k4')

if not os.path.exists(imagenet_directory + '/train10k5'):
    os.makedirs(imagenet_directory + '/train10k5')


for c in classes:
    if not os.path.exists('{}/train10k3/{}'.format(imagenet_directory, c)):
        os.makedirs('{}/train10k3/{}'.format(imagenet_directory, c))
    if not os.path.exists('{}/train10k4/{}'.format(imagenet_directory, c)):
        os.makedirs('{}/train10k4/{}'.format(imagenet_directory, c))
    if not os.path.exists('{}/train10k5/{}'.format(imagenet_directory, c)):
        os.makedirs('{}/train10k5/{}'.format(imagenet_directory, c))


for s in sets:
    for c in classes:
        source_directory = '{}/{}/{}'.format(cinic_directory, s, c)
        filenames = glob.glob('{}/*.png'.format(source_directory))
        random.shuffle(filenames)
        # filenames1 = filenames[0:1000]
        # for fn in filenames1:
        #     dest_fn = fn.split('/')[-1]
        #     dest_fn = '{}/train10k/{}/{}'.format(imagenet_directory, c, dest_fn)
        #     if symlink:
        #         if not os.path.islink(dest_fn):
        #             os.symlink(fn, dest_fn)
        #     else:
        #         copyfile(fn, dest_fn)
        # filenames2 = filenames[1000:2000]
        # for fn in filenames2:
        #     dest_fn = fn.split('/')[-1]
        #     dest_fn = '{}/valid10k/{}/{}'.format(imagenet_directory, c, dest_fn)
        #     if symlink:
        #         if not os.path.islink(dest_fn):
        #             os.symlink(fn, dest_fn)
        #     else:
        #         copyfile(fn, dest_fn)
        filenames3 = filenames[2000:3000]
        for fn in filenames3:
            dest_fn = fn.split('/')[-1]
            dest_fn = '{}/train10k3/{}/{}'.format(imagenet_directory, c, dest_fn)
            if symlink:
                if not os.path.islink(dest_fn):
                    os.symlink(fn, dest_fn)
            else:
                copyfile(fn, dest_fn)
        filenames4 = filenames[3000:4000]
        for fn in filenames4:
            dest_fn = fn.split('/')[-1]
            dest_fn = '{}/train10k4/{}/{}'.format(imagenet_directory, c, dest_fn)
            if symlink:
                if not os.path.islink(dest_fn):
                    os.symlink(fn, dest_fn)
            else:
                copyfile(fn, dest_fn)
        filenames5 = filenames[4000:5000]
        for fn in filenames5:
            dest_fn = fn.split('/')[-1]
            dest_fn = '{}/train10k5/{}/{}'.format(imagenet_directory, c, dest_fn)
            if symlink:
                if not os.path.islink(dest_fn):
                    os.symlink(fn, dest_fn)
            else:
                copyfile(fn, dest_fn)

