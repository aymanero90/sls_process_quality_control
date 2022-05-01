#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

tf.config.get_visible_devices()


# In[ ]:


image_train_gen = keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.2,
                                                         height_shift_range=0.2,
                                                         rescale=1 / 255.,
                                                         horizontal_flip=True,
                                                         fill_mode="nearest",
                                                         validation_split=0.2
                                                         )

image_test_gen = keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.2,
                                                         height_shift_range=0.2,
                                                         rescale=1 / 255.,
                                                         horizontal_flip=True,
                                                         fill_mode="nearest"
                                                         )


# In[ ]:


train_images = image_train_gen.flow_from_directory(
    directory="scanner/train/",
    color_mode="rgb",
    class_mode="binary",
    subset='training',
    seed=42
    )


# In[ ]:


valid_images = image_train_gen.flow_from_directory(
    directory="scanner/train/",
    color_mode="rgb",
    class_mode="binary",
    subset='validation',
    seed=42
    )


# In[ ]:


test_images = image_test_gen.flow_from_directory(
    directory="scanner/test/",
    color_mode="rgb",
    class_mode="binary",
    seed=42
    )


# In[ ]:


train_images.image_shape


# In[ ]:


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[256,256,3]))
model.add(keras.layers.Dense(1000, activation="relu"))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))  # "sigmoid" for binary classification


# In[ ]:


model.compile(loss="binary_crossentropy",   ## for binary classification
              optimizer="sgd",
              metrics=["accuracy"])


# In[ ]:


early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
checkpoint_cb = keras.callbacks.ModelCheckpoint("models/model-{epoch:02d}-{val_accuracy:.2f}.hdf5",  monitor='val_accuracy', save_best_only=True)


# In[ ]:


history = model.fit(train_images, epochs=100, validation_data=valid_images, callbacks=[early_stopping_cb, checkpoint_cb])


# In[ ]:




