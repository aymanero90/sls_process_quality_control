import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import shutil
from tensorflow import keras
from tqdm import tqdm


# Define paths

LASER_MODEL_PATH = 'models/laser.hdf5'
SCRAPER_MODEL_PATH = 'models/scraper.hdf5'
SOURCE_IMAGES_PATH = "images/*.jpg"
SCRAPER_LASER_IMG_PATH = 'scrapers_lasers_images/'


# Load the classifiers

scraperClassifier = keras.models.load_model(SCRAPER_MODEL_PATH)
laserClassifier = keras.models.load_model(LASER_MODEL_PATH)


# Define help and methods

def classify_image(image , classifier):
    image = cv2.resize(image , (256, 256))
    img_list = [image]
    img_array = np.asarray(img_list)
    prediction = classifier.predict(img_array)
    return prediction


def image_in_scraper_laser_phase(img):
    ret = False
    if classify_image(img , scraperClassifier) == 1 or classify_image(img , laserClassifier) == 1:
        ret = True
    return ret


# Read and filter images from scraper and laser phases

path = glob.glob(SOURCE_IMAGES_PATH)
for file in tqdm(path):
    img = plt.imread(file)    
    if image_in_scraper_laser_phase(img):
        shutil.move(file, SCRAPER_LASER_IMG_PATH)
    else:
        img = img [30:300]     # image cleaning from the dark side at top and the logo at bottom
        plt.imsave(file, img)  # replace the old image with the cropped one.