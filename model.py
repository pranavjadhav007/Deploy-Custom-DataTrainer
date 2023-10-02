# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 17:38:47 2023

@author: prana
"""
import keras
import tensorflow
from tensorflow.keras.preprocessing import image
from keras.models import load_model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model



model = VGG19(weights='imagenet',include_top=False,
       input_shape=(180, 180, 3))
model.save('static/model_VGG19.h5')
