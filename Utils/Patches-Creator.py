#!/usr/bin/env python
# coding: utf-8

import cv2
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

BASIC_PATH = "./images/"
IMG_SIZE = 240


def create_image_patches(img_path, target_path, n_patches):
    img = plt.imread(img_path)
    img = img[40:290,:,:]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    patch_size = int(IMG_SIZE / n_patches)
    k = 0
    for i in range(0, IMG_SIZE, patch_size):
        for j in range(0, IMG_SIZE, patch_size):
            patch = img[i : i + patch_size, j : j + patch_size , :]
            index = img_path.rfind("\\")
            if index == -1:
                index = img_path.rfind("/")
            file_name = target_path + file[index + 1: -4] + "_" + str(k) + ".jpg"
            plt.imsave(file_name, patch)
            k += 1

path = glob.glob(BASIC_PATH + "*.jpg")
n_patches = 4
target_path = BASIC_PATH + "patches/"
if IMG_SIZE % n_patches != 0:
    raise Exception("Wrong patches number!!")
for file in tqdm(path):
     create_image_patches(file, target_path, n_patches)