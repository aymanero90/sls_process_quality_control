#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm

BASIC_PATH = "/images/"
IMAGE_WIDTH = 60
IMAGE_HEIGHT = 70


def cluster_and_arrange_image(img, n_clusters):
    X = img.reshape(-1,3)
    km = KMeans(n_clusters=n_clusters).fit(X)
    means = np.mean(km.cluster_centers_, axis=1)
    sorted_means = np.sort(means)
    n_pixels_per_cluster = int(IMAGE_HEIGHT * IMAGE_WIDTH / n_clusters)
    idxs = []
    for i in range(n_clusters):
        for j in range(n_clusters):
            if sorted_means[i] == means[j]:
                idxs.append(j)
                break
    segmented_arranged_img = []
    for i in range (n_clusters):
        cluster = X[km.labels_ == idxs[i]]
        if len(cluster) < n_pixels_per_cluster :
            ix = idxs[i]
            for i in range(len(cluster), n_pixels_per_cluster):
                cluster = np.append(cluster, km.cluster_centers_[ix:ix + 1], axis=0)
        else:
            cluster = cluster[:n_pixels_per_cluster]
        segmented_arranged_img.extend(np.uint8(cluster))
    segmented_arranged_img =  np.asarray(segmented_arranged_img)
    segmented_arranged_img = segmented_arranged_img.reshape(img.shape)
    return segmented_arranged_img , km.cluster_centers_


def read_and_cluster_images(src, dest, width, height):
    path = glob.glob(src)
    for i, file in enumerate(tqdm(path)):
        img = plt.imread(file)
        img = img[30:300,:,:]
        img = cv2.resize(img, (width,height))
        clustered_image, centers = cluster_and_arrange_image(img, 3)
        index = file.rfind("\\")
        if index == -1:
            index = file.rfind("/")
        file_name = file[index + 1:]
        plt.imsave(dest + "_clustered_"+ file_name , clustered_image)
#         np.save(dest + "clusters/" + file_name, centers)



read_and_cluster_images(BASIC_PATH + "*.jpg"   ,BASIC_PATH + "images/clustered/", IMAGE_WIDTH, IMAGE_HEIGHT)

