# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:46:24 2023

@author: prana
"""
import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import pathlib
from tensorflow.keras.optimizers import Adam



data_dir = "C:/Users/prana/CustomDataTrainer/FolderTrainer/Training"
val_dir= "C:/Users/prana/CustomDataTrainer/FolderTrainer/Validation"

img_height,img_width=180,180
batch_size=32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

print("Classes loaded")
class_names = train_ds.class_names
no_of_classes=len(class_names)
print("Model build Start")

model = Sequential()

pretrained_model= tf.keras.applications.VGG16(include_top=False,
                   input_shape=(180,180,3),
                   pooling='avg',classes=no_of_classes,
                   weights='imagenet')
for layer in pretrained_model.layers[:-23]:
    layer.trainable=False

model.add(pretrained_model)
model.add(Flatten())
model.add(Dense(no_of_classes, activation='softmax'))



# MODEL_PATH = 'static/model_VGG19.h5'
# model = load_model(MODEL_PATH)
# base_model=model
# base_model.trainable = False
# num_classes = no_of_classes
# model = Sequential([
# base_model,
# Flatten(),
# Dense(num_classes, activation='softmax')
# ])

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

wt= model.weights[1]
epochs=1
history = model.fit(train_ds,validation_data=val_ds,epochs=epochs)


# model.save('models/model_VGG19.h5')
# with open("models/model.pkl",'wb') as files:
#   pickle.dump(model,files)
model.save("my_model.keras")
