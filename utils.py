import math
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import warnings
import matplotlib.image as imgmp
from keras import backend as K
from keras.preprocessing import image

def rename_directory(images_directory,annotations_directory):
    races_list = os.listdir(images_directory)
    annotation_list = os.listdir(annotations_directory)
    for race in races_list:
        if(race.startswith("n02")):
            oldname = race
            new_name = race.split('-',1)[1]
            os.rename(images_directory+'/'+race, images_directory+'/'+new_name)

    for race in annotation_list:
        if(race.startswith("n02")):
            oldname = race
            new_name = race.split('-',1)[1]
            os.rename(annotations_directory+'/'+race, annotations_directory+'/'+new_name)
    return True

def show_images_from_directory(path, classes, num_sample):
    fig = plt.figure(figsize=(20,20))
    fig.patch.set_facecolor('#377AB7')
    plt.suptitle("{}".format(classes), y=.83,color="white", fontsize=22)
    images = os.listdir(path + "/" + classes)[:num_sample]
    for i in range(num_sample):
        img = imgmp.imread(path+"/"+classes+"/"+images[i])
        plt.subplot(num_sample/num_sample+1, num_sample, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()  

def show_images_sample(contents_images,num_sample=4):
  fig = plt.figure(figsize=(20,20))
  for i in range(num_sample):
      r = np.random.randint(0, len(contents_images))
      plt.subplot(num_sample/num_sample+1, num_sample, i+1)
      plt.imshow(image.array_to_img(contents_images[r]))
      plt.axis('off')
  plt.show()  


def plot_image(i, predictions_array, true_label, img, class_names):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(image.array_to_img(img), cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label,num_class):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(num_class))
  plt.yticks([])
  thisplot = plt.bar(range(num_class), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

def plot_predictions(x_test,y_test,Y_pred,class_names,num_class):
  num_rows = 5
  num_cols = 2
  num_images = num_rows*num_cols
  plt.figure(figsize=(8*num_cols, 2*num_rows))
  for i in range(num_images):
    r = int(np.random.randint(0, y_test.shape[0], 1))
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(r, Y_pred[r], y_test, x_test,class_names)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(r, Y_pred[r], y_test,num_class)
  plt.tight_layout()
  plt.show()  

# Create histogram
def plot_histogram(init_img, convert_img):
    """Function allowing to display the initial
    and converted images according to a certain
    colorimetric format as well as the histogram
    of the latter. 

    Parameters
    -------------------------------------------
    init_img : list
        init_img[0] = Title of the init image
        init_img[1] = Init openCV image
    convert_img : list
        convert_img[0] = Title of the converted
        convert_img[1] = converted openCV image
    -------------------------------------------
    """
    hist, bins = np.histogram(
                    convert_img[1].flatten(),
                    256, [0,256])
    # Cumulative Distribution Function
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    # Plot histogram
    fig = plt.figure(figsize=(25,6))
    plt.subplot(1, 3, 1)
    plt.imshow(init_img[1])
    plt.title("{} Image".format(init_img[0]), 
              color="#343434")
    plt.subplot(1, 3, 2)
    plt.imshow(convert_img[1])
    plt.title("{} Image".format(convert_img[0]), 
              color="#343434")
    plt.subplot(1, 3, 3)
    plt.plot(cdf_normalized, 
             color='r', alpha=.7,
             linestyle='--')
    plt.hist(convert_img[1].flatten(),256,[0,256])
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.title("Histogram of convert image", color="#343434")
    plt.suptitle("Histogram and cumulative "\
                 "distribution for test image",
              color="black", fontsize=22, y=.98)
    plt.show()    

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