#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import glob
from tqdm import tqdm
from tensorflow import keras
import shutil
import matplotlib.pyplot as plt

THRESHOLD = 0.09
scraper_classifier = keras.models.load_model("../models/scraper-classifier-new.hdf5")
laser_classifier = keras.models.load_model("../models/laser-classifier-new.hdf5")

path = glob.glob("../images/*.jpg")
for file in tqdm(path):
    img = plt.imread(file)
    img = img[30: -30]
    img = cv2.resize(img, (256, 256))
    img = img / 255.
    img = np.asarray([img])

    if scraper_classifier.predict(img) > THRESHOLD:
        shutil.move(file, "../images/scraper/")

    elif laser_classifier.predict(img) > THRESHOLD:
        shutil.move(file, "../images/laser/")






