# -*- coding: utf-8 -*-
"""
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
from tensorflow.keras.preprocessing import image
import cv2
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from keras.preprocessing import image


app=Flask(__name__)

MODEL_PATH = 'static/runtime_model.h5'

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/usecloud")
def usecloud():
    return render_template("use_cloud.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/uselocal")
def uselocal():
    return render_template("use_local.html")

data_dir = "C:/Users/prana/CustomDataTrainer/FolderTrainer/Training"
val_dir= "C:/Users/prana/CustomDataTrainer/FolderTrainer/Validation"

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    image=cv2.imread(str(file_path))
    image_resized= cv2.resize(image, (180,180))
    image=np.expand_dims(image_resized,axis=0)
    
        
    classcheck= tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.1,
        subset="training",
        seed=123,
        image_size=(180, 180),
        batch_size=32)
        
    class_names = classcheck.class_names
    no_of_classes=len(class_names)
    print("Classes loaded")
    print("Model build Start")    
    runtime_model = load_model(MODEL_PATH)

    inps=[request.form.get('class_category'),request.form.get('drive_link')]
            
    pred=runtime_model.predict(image)
    result=class_names[np.argmax(pred)] 
    
    return '<span class="final_call">Results and info of Trained Model</span><br><br>Number of classes found: </b>'+str(no_of_classes)+" \n<b> Categories:</b> "+str(class_names)+"\n<b> Output Class: </b>"+str(result)+'\n <img class="grpahs" src="static/model_accuracy.png" alt=""> <img class="grpahs" src="static/model_loss.png" alt=""> '


    # return render_template('index.html',noofclasses="Number of classes found are: "+str(no_of_classes),
    #                        nameofclasses="Categories: "+str(class_names),
    #                        output="Result: "+str(result))

if __name__ =="__main__":
    app.run(debug=True)
    
     


