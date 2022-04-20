from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.applications.xception import preprocess_input
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as imgmp
import seaborn as sns
import timeit
import cv2 as cv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix 
import sys
from IPython.display import display
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.models import load_model
import json
import os, logging 

logging.disable(logging.WARNING) 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Save it under the form of a json file
class_names = json.load(open('./dogs_classification/modele/class_names', 'r'))

# Metrics have been removed from Keras core. We need to calculate them manually
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

my_xcept_model = load_model('./dogs_classification/modele/my_xcept_model_all_tune.h5', custom_objects={"f1_m": f1_m})



def predict_image(imagePath):
    #load the image
    display(image.load_img(imagePath, target_size = (299, 299)))
    my_image = load_img(imagePath, target_size=(299, 299))
    
    # Convert to RGB
    src = cv.imread(imagePath)
    img = cv.cvtColor(src,cv.COLOR_BGR2RGB)
    dim = (299, 299)
    img = cv.resize(img, dim, interpolation=cv.INTER_LINEAR)
    # Equalization
    img_yuv = cv.cvtColor(img,cv.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])
    img_equ = cv.cvtColor(img_yuv, cv.COLOR_YUV2RGB)
    # Apply non-local means filter on test img
    dst_img = cv.fastNlMeansDenoisingColored(
        src=img_equ,
        dst=None,
        h=15,
        hColor=15,
        templateWindowSize=7,
        searchWindowSize=21)
    my_image = img_to_array(dst_img)

    #preprocess the image
    my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
    my_image = preprocess_input(my_image)

    #make the prediction
    prediction = my_xcept_model.predict(my_image)

    # print([np.round(x) for x in prediction])
    predicted_class=prediction[0].argmax() 
    sorting = (-prediction).argsort()

    # getting the top 2 predictions
    sorted_ = sorting[0][:3]

    for value in sorted_:
        # you can get your classes from the encoder(your_classes = encoder.classes_) 
        # or from a dictionary that you created before.
        # And then we access them with the predicted index.
        predicted_label = class_names[value]

        # just some rounding steps
        prob = (prediction[0][value]) * 100
        prob = "%.2f" % round(prob,2)
        print("confidence %s%% sure that it belongs to %s." % (prob, predicted_label))
        

if __name__ == "__main__":
    args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
    globals()[args[1]](*args[2:])
