#!/usr/bin/env python
# coding: utf-8

from tensorflow import keras
import tensorflow as tf


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


train_images = image_train_gen.flow_from_directory(
    directory="laser/train/",
    color_mode="rgb",
    class_mode="binary",
    subset='training',
    seed=42
    )


valid_images = image_train_gen.flow_from_directory(
    directory="laser/train/",
    color_mode="rgb",
    class_mode="binary",
    subset='validation',
    seed=42
    )

test_images = image_test_gen.flow_from_directory(
    directory="laser/test/",
    color_mode="rgb",
    class_mode="binary",
    seed=42
    )


## building the model

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[256,256,3]))
model.add(keras.layers.Dense(1000, activation="relu"))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))  # "sigmoid" for binary classification


model.compile(loss="binary_crossentropy",   ## for binary we use "binary_crossentropy"
              optimizer="sgd",    ###  optimizer=keras.optimizers.SGD(lr=???) for learnin grate 
              metrics=["accuracy"])


model.summary()

early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
checkpoint_cb = keras.callbacks.ModelCheckpoint("models/model-{epoch:02d}-{val_accuracy:.2f}.hdf5",  monitor='val_accuracy', save_best_only=True)


history = model.fit(train_images, epochs=100, validation_data=valid_images, callbacks=[early_stopping_cb, checkpoint_cb])

