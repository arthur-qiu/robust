import numpy as np
from PIL import Image
import os

file = '/Users/qiuhaonan/Desktop/temp/samples.npz'
save = '/Users/qiuhaonan/Desktop/image_dataset/cinic-10-cifar/gan50k_s0/'

# file = '/home/xiaocw/haonan/BigGAN-PyTorch/samples/BigGAN_C10_seed0_Gch64_Dch64_bs32_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema/samples.npz'
# save = '/home/xiaocw/haonan/cinic-10-cifar/gan150k_s0/'

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

if not os.path.exists(save):
    os.makedirs(save)

for class_name in classes:
    if not os.path.exists(save + class_name):
        os.makedirs(save + class_name)

npz_file = np.load(file)

x = npz_file['x']
y = npz_file['y']

count = 0
for i in range(50000):
    count += 1
    img = x[i]
    label = y[i]

    final_image = Image.fromarray(img.transpose(1,2,0))

    final_image.save(save + classes[label]+ '/' + str(count)+ '.png')

