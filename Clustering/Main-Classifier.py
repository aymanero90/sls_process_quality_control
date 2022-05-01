#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tensorflow import keras
import tensorflow as tf

BASIC_PATH = "./images/"

tf.config.get_visible_devices()


# ## Define ImageDataGenerators with image preprocessing <br> and  read train, validation and test images

def get_images(path):
    image_train_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.,
                                                             horizontal_flip=True,
                                                             fill_mode="nearest",
                                                             validation_split=0.25
                                                             )
    
    image_test_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.,
                                                             horizontal_flip=True,
                                                             fill_mode="nearest"
                                                             )
   
    
    train_images = image_train_gen.flow_from_directory(
       directory= path + "/train/",
       color_mode="rgb",
       class_mode="binary",
       subset='training',
       target_size=(70, 60),
       seed=42
       )
    
    valid_images = image_train_gen.flow_from_directory(
       directory= path + "/train/",
       color_mode="rgb",
       class_mode="binary",
       subset='validation',
       target_size=(70, 60),
       seed=42
       )
    
    test_images = image_test_gen.flow_from_directory(
       directory= path + "/test/",
       color_mode="rgb",
       class_mode="binary",
       target_size=(70, 60),
       seed=42
       )
    
    return train_images, valid_images, test_images




train_images, valid_images, test_images = get_images(BASIC_PATH +"images/experiment-classifier/clustered-3-improvement/")


# ## Create the model with 3 hidden layers and one binary output layer

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[70, 60,3]))
model.add(keras.layers.Dense(700, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))  # "sigmoid" activation function for binary classification


model.compile(loss="binary_crossentropy",   ## for binary classification
              optimizer="sgd",
              metrics=["accuracy"])


early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
checkpoint_cb = keras.callbacks.ModelCheckpoint("models/Main-Classifier-3-clusters-improved-{epoch:02d}-{val_accuracy:.2f}.hdf5",  monitor='val_accuracy', save_best_only=True)


# ## Train the model

history = model.fit(train_images, epochs=100, validation_data=valid_images, callbacks=[early_stopping_cb, checkpoint_cb])


# ## evaluate the test dataset with the best trained model

model.evaluate(test_images)
model.summary()





