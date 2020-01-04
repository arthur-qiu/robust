from PIL import Image
import numpy as np
import os
import random
random.seed(1)

path1 = '/Users/qiuhaonan/Desktop/image_dataset/cinic-10-cifar/train10k4/'
path2 = '/Users/qiuhaonan/Desktop/image_dataset/cinic-10-cifar/train10k4_combine/'
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

name_list = []
for i in range(len(classes)):
    old_image_directory = path1 + classes[i]
    image_directory = path2 + classes[i]
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    names = os.listdir(old_image_directory)
    names.sort()
    name_list.append(names)

for i in range(len(classes)):
    number_imgs = len(name_list[i])
    for j in range(number_imgs):
        ran_folder = (random.randint(1,9) + i) % 10

        img_path1 = path1 + classes[i] + '/'+ name_list[i][j]
        img_path2 = path1 + classes[ran_folder] + '/'+ name_list[ran_folder][(i + j + ran_folder) % number_imgs]

        img1 = np.array(Image.open(img_path1))
        img2 = np.array(Image.open(img_path2))

        img3 = (img1/2 + img2/2).astype(np.uint8)

        final_image = Image.fromarray(img3)

        final_image.save(path2 + classes[i] + '/'+ name_list[i][j])

