import os
from flask import Flask
from flask import request
from flask import render_template
import os,glob
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import classification_report
import cv2
import numpy as np
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf

app =Flask(__name__)
UPLOAD_FOLDER = "\ACM\ACMProject\PROJECT\webapp\static"
MODEL=None
DEVICE="cuda"
@app.route("/",methods=["GET"])
# def upload_predict():
#     if request.method == "POST":
#         image_file = request.files["image"]
#         if image_file:
#             image_location = os.path.join(
#                 UPLOAD_FOLDER,
#                 image_file.filename
#             )
#             image_file.save(image_location)
#             return render_template("index.html",prediction=1)
#     return render_template("index.html",prediction=0)
@app.route("/",methods=["POST"])
def upload_predict():
    image_file = request.files["image"]
    image_path = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
    image_file.save(image_path)
    image=load_img(image_path,target_size=(224,224))
    image.reshape()
            
    return render_template("index.html")
    
if __name__ == "__main__":
    app.run(port=12000, debug=True)