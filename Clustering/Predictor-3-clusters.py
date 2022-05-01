#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import shutil
from tensorflow import keras
from tqdm import tqdm
from keras.metrics import mean_squared_error as mse




BASIC_PATH = "./images/"
THRESHOLD = 2600
GREEN_THRESHOLD = 25
IMAGE_WIDTH = 60
IMAGE_HEIGHT = 70


# ## Load the classifiers

scraperClassifier = keras.models.load_model(BASIC_PATH + 'models/ScraperClassifier.hdf5')
laserClassifier = keras.models.load_model(BASIC_PATH + 'models/LaserClassifier.hdf5')
MainClassifier = keras.models.load_model(BASIC_PATH + 'models/MainClassifier.hdf5')

MainClassifier.summary()


# ## Define help and clustering methods

def classify_image(image , classifier):
    image = cv2.resize(image , (256,256))
    img = image
    img_list = []
    img_list.append(img)
    img_array = np.asarray(img_list)
    predection = classifier.predict(img_array)
    return predection


def image_in_scraper_phase(img):
    ret = False
    if classify_image(img , scraperClassifier) == 1:
        ret = True
    return ret


def image_in_laser_phase(img):
    ret = False
    if classify_image(img , laserClassifier) == 1:
        ret = True
    return ret


def cluster_image(img, n_clusters):
    X = img.reshape(-1,3)
    km = KMeans(n_clusters=n_clusters).fit(X)
    means = np.mean(km.cluster_centers_, axis=1)
    sorted_means = np.sort(means)
    sorted_clusters = []
    for i in range(0,n_clusters):
        for j in range(0,n_clusters):
            if sorted_means[i] == means[j]:
                sorted_clusters.append(km.cluster_centers_[j])
                break
    return np.int64(sorted_clusters)


def image_has_defect_part(img):
    anomaly = False
    img = img[30:300,:,:]          ## image cleaning from the dark side at top and the logo at bottom
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    
    centers= cluster_image(img, 3)
    
    mean_squared_error =  mse(centers[0], centers[1])
    if mean_squared_error > THRESHOLD :
        anomaly = True
    else:
        new_centers = cluster_image(img, 20)
        g_channel = new_centers[0][1]
        if g_channel < GREEN_THRESHOLD:    ## check the green channel of the first cluster
            anomaly = True
        else:
            anomaly = False
    return anomaly


def get_clustered_image(img, n_clusters, img_width, img_heigth):
    X = img.reshape(-1,3)
    km = KMeans(n_clusters= n_clusters).fit(X)
    n_pixels_per_cluster = int(img_heigth * img_width / n_clusters)
    means = np.mean(km.cluster_centers_, axis=1)
    sorted_means = np.sort(means)
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
    return segmented_arranged_img    




# ## Way 1: Read and predict images with threshold



TARGET_PATH = "images/1new/2/"

path = glob.glob(BASIC_PATH +"images/1new/2/source/*.jpg")

for file in tqdm(path):
    img = plt.imread(file)
    
    if image_in_scraper_phase(img):
        shutil.move(file, BASIC_PATH + TARGET_PATH + "scraper/")
        continue
    
    if image_in_laser_phase(img):
        shutil.move(file, BASIC_PATH + TARGET_PATH + "laser/")
        continue
    
    if image_has_defect_part(img):
        shutil.move(file, BASIC_PATH + TARGET_PATH + "fail/")
    else:
        shutil.move(file, BASIC_PATH + TARGET_PATH + "normal/")




# ## Way 2: Read and predict images with Classifier



TARGET_PATH = "images/test-normal-images/"
n_clusters = 3

path = glob.glob(BASIC_PATH +"images/experiment-classifier/sample/*.jpg")

if IMAGE_HEIGHT * IMAGE_WIDTH % n_clusters != 0:
     raise Exception("Wrong number of clusters !!!")
for i, file in enumerate(tqdm(path)):
    img = plt.imread(file)    
    if image_in_scraper_phase(img):
        print("scraper")        
        continue
 
    if image_in_laser_phase(img):
        print("laser")
        continue

    
    img = cv2.resize(img, (240, 320))
    img = img[30:280,:,:] 
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    clustered_image = get_clustered_image (img, n_clusters, IMAGE_WIDTH, IMAGE_HEIGHT)
    clustered_image = np.asarray(clustered_image)
    plt.imsave(file + str(i) + ".jpg",clustered_image )
    image_list = [clustered_image]
    
    if MainClassifier.predict(np.asarray(image_list)) == 1:
        print("fail: " +  file)
    else:
        print("normal: " + file)




