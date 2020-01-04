import scipy
import scipy.io as scio
import os
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

a = unpickle('/Users/qiuhaonan/Desktop/image_dataset/imagenet32/val_data')
imgs = a['data']
print(imgs.shape)

folder = '/Users/qiuhaonan/Desktop/image_dataset/imagenet32/100img/'

for i in range(1000):
    name = folder + str(i) + '.png'
    img = imgs[i].reshape(3,32,32)
    img = img.transpose(1,2,0)
    # print(img.shape)
    scipy.misc.imsave(name, img)

# import scipy
# import scipy.io as scio
# import os
# import numpy as np
#
# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo)
#     return dict
#
# a = unpickle('/Users/qiuhaonan/Desktop/image_dataset/cifar/cifar-10-batches-py/test_batch')
# imgs = a['data']
# print(imgs.shape)
#
# folder = '/Users/qiuhaonan/Desktop/image_dataset/cifar/cifar10_100img/'
#
# for i in range(1000):
#     name = folder + str(i) + '.png'
#     img = imgs[i].reshape(3,32,32)
#     img = img.transpose(1,2,0)
#     # print(img.shape)
#     scipy.misc.imsave(name, img)