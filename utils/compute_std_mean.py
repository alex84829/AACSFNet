import numpy as np
import cv2
import random
import os


filepath = r"../data/data_orig/UNBC/cropped_ImagesExp"# 数据集目录

means = [0, 0, 0]
stdevs = [0, 0, 0]
num_imgs = 0

for root, dirs, files in os.walk(filepath):
    for file in files:
        img_path = os.path.join(root, file)

        num_imgs += 1

        img = cv2.imread(img_path)
        img = np.asarray(img)

        img = img.astype(np.float32) / 255.
        for i in range(3):
            means[i] += img[:, :, i].mean()
            stdevs[i] += img[:, :, i].std()

# cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
print(num_imgs)
means.reverse()     # BGR --> RGB
stdevs.reverse()

means = np.asarray(means) / num_imgs
stdevs = np.asarray(stdevs) / num_imgs

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))