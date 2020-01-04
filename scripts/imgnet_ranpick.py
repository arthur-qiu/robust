import os
import glob
import numpy as np
from shutil import copyfile
symlink = False    # If this is false the files are copied instead
combine_train_valid = False    # If this is true, the train and valid sets are ALSO combined
combine_all = False
import random
random.seed(1)

cinic_directory = "/Users/qiuhaonan/Desktop/image_dataset/cinic_imgnet"
imagenet_directory = "/Users/qiuhaonan/Desktop/image_dataset/cinic_imgnetdivide"
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# sets = ['train', 'valid', 'test']
sets = ['train', 'test']
# if not os.path.exists(imagenet_directory):
#     os.makedirs(imagenet_directory)
# if not os.path.exists(imagenet_directory + '/train'):
#     os.makedirs(imagenet_directory + '/train')
# if not os.path.exists(imagenet_directory + '/test'):
#     os.makedirs(imagenet_directory + '/test')
#
# for c in classes:
#     if not os.path.exists('{}/train/{}'.format(imagenet_directory, c)):
#         os.makedirs('{}/train/{}'.format(imagenet_directory, c))
#     if not combine_all:
#         if not os.path.exists('{}/test/{}'.format(imagenet_directory, c)):
#             os.makedirs('{}/test/{}'.format(imagenet_directory, c))
#     if not combine_train_valid and not combine_all:
#         if not os.path.exists('{}/valid/{}'.format(imagenet_directory, c)):
#             os.makedirs('{}/valid/{}'.format(imagenet_directory, c))

if not os.path.exists(imagenet_directory):
    os.makedirs(imagenet_directory)
if not os.path.exists(imagenet_directory + '/train10k'):
    os.makedirs(imagenet_directory + '/train10k')
if not os.path.exists(imagenet_directory + '/test10k'):
    os.makedirs(imagenet_directory + '/test10k')
if not os.path.exists(imagenet_directory + '/train40k'):
    os.makedirs(imagenet_directory + '/train40k')
if not os.path.exists(imagenet_directory + '/test40k'):
    os.makedirs(imagenet_directory + '/test40k')

for c in classes:
    if not os.path.exists('{}/train40k/{}'.format(imagenet_directory, c)):
        os.makedirs('{}/train40k/{}'.format(imagenet_directory, c))
    if not combine_all:
        if not os.path.exists('{}/test40k/{}'.format(imagenet_directory, c)):
            os.makedirs('{}/test40k/{}'.format(imagenet_directory, c))
    if not os.path.exists('{}/train10k/{}'.format(imagenet_directory, c)):
        os.makedirs('{}/train10k/{}'.format(imagenet_directory, c))
    if not combine_all:
        if not os.path.exists('{}/test10k/{}'.format(imagenet_directory, c)):
            os.makedirs('{}/test10k/{}'.format(imagenet_directory, c))
    # if not combine_train_valid and not combine_all:
    #     if not os.path.exists('{}/valid/{}'.format(imagenet_directory, c)):
    #         os.makedirs('{}/valid/{}'.format(imagenet_directory, c))

for s in sets:
    for c in classes:
        source_directory = '{}/{}/{}'.format(cinic_directory, s, c)
        filenames = glob.glob('{}/*.png'.format(source_directory))
        random.shuffle(filenames)
        filenames1 = filenames[0:1000]
        for fn in filenames1:
            dest_fn = fn.split('/')[-1]
            if (s == 'train' or s == 'valid' or s == 'test') and combine_all and 'cifar' not in fn.split('/')[-1]:
                dest_fn = '{}/train/{}/{}'.format(imagenet_directory, c, dest_fn)
                if symlink:
                    if not os.path.islink(dest_fn):
                        os.symlink(fn, dest_fn)
                else:
                    copyfile(fn, dest_fn)

            elif (s == 'train' or s == 'valid') and combine_train_valid and 'cifar' not in fn.split('/')[-1]:
                dest_fn = '{}/train/{}/{}'.format(imagenet_directory, c, dest_fn)
                if symlink:
                    if not os.path.islink(dest_fn):
                        os.symlink(fn, dest_fn)
                else:
                    copyfile(fn, dest_fn)

            elif (s == 'train') and 'cifar' not in fn.split('/')[-1]:
                dest_fn = '{}/train10k/{}/{}'.format(imagenet_directory, c, dest_fn)
                if symlink:
                    if not os.path.islink(dest_fn):
                        os.symlink(fn, dest_fn)
                else:
                    copyfile(fn, dest_fn)

            elif (s == 'valid') and 'cifar' not in fn.split('/')[-1]:
                dest_fn = '{}/valid/{}/{}'.format(imagenet_directory, c, dest_fn)
                if symlink:
                    if not os.path.islink(dest_fn):
                        os.symlink(fn, dest_fn)
                else:
                    copyfile(fn, dest_fn)

            elif s == 'test' and 'cifar' not in fn.split('/')[-1]:
                dest_fn = '{}/test10k/{}/{}'.format(imagenet_directory, c, dest_fn)
                if symlink:
                    if not os.path.islink(dest_fn):
                        os.symlink(fn, dest_fn)
                else:
                    copyfile(fn, dest_fn)
        filenames2 = filenames[1000:5000]
        for fn in filenames2:
            dest_fn = fn.split('/')[-1]
            if (s == 'train' or s == 'valid' or s == 'test') and combine_all and 'cifar' not in fn.split('/')[-1]:
                dest_fn = '{}/train/{}/{}'.format(imagenet_directory, c, dest_fn)
                if symlink:
                    if not os.path.islink(dest_fn):
                        os.symlink(fn, dest_fn)
                else:
                    copyfile(fn, dest_fn)

            elif (s == 'train' or s == 'valid') and combine_train_valid and 'cifar' not in fn.split('/')[-1]:
                dest_fn = '{}/train/{}/{}'.format(imagenet_directory, c, dest_fn)
                if symlink:
                    if not os.path.islink(dest_fn):
                        os.symlink(fn, dest_fn)
                else:
                    copyfile(fn, dest_fn)

            elif (s == 'train') and 'cifar' not in fn.split('/')[-1]:
                dest_fn = '{}/train40k/{}/{}'.format(imagenet_directory, c, dest_fn)
                if symlink:
                    if not os.path.islink(dest_fn):
                        os.symlink(fn, dest_fn)
                else:
                    copyfile(fn, dest_fn)

            elif (s == 'valid') and 'cifar' not in fn.split('/')[-1]:
                dest_fn = '{}/valid/{}/{}'.format(imagenet_directory, c, dest_fn)
                if symlink:
                    if not os.path.islink(dest_fn):
                        os.symlink(fn, dest_fn)
                else:
                    copyfile(fn, dest_fn)

            elif s == 'test' and 'cifar' not in fn.split('/')[-1]:
                dest_fn = '{}/test40k/{}/{}'.format(imagenet_directory, c, dest_fn)
                if symlink:
                    if not os.path.islink(dest_fn):
                        os.symlink(fn, dest_fn)
                else:
                    copyfile(fn, dest_fn)