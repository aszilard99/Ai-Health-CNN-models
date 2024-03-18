from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D,Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16
import h5py
import os
import random
import cv2
import imutils
import numpy as np
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
import seaborn as sns
import pandas as pd

from os import listdir

def build_vgg16extended_model_figshare(input_shape):

  conv = VGG16(input_shape= input_shape, weights='imagenet',include_top=False)

  for layer in conv.layers:
    layer.trainable = False

  x = conv.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024,activation='relu')(x)
  x = Dense(1024,activation='relu')(x)
  x = Dense(512, activation='relu')(x)
  x = Dropout(.2)(x)
  pred = Dense(3,activation='softmax')(x)
  model = Model(inputs=conv.input, outputs=pred, name='VGG16Extended')

  model.summary()

  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  return model

def build_vgg16_model_figshare(input_shape):

    conv = VGG16(input_shape=input_shape, weights='imagenet', include_top=False)

    for layer in conv.layers:
        layer.trainable = False

    x = conv.output
    x = Flatten()(x)
    x = Dense(units=4096,activation="relu")(x)
    x = Dense(units=4096,activation="relu")(x)
    pred = Dense(units=3, activation="softmax")(x)
    model = Model(inputs=conv.input, outputs=pred, name='VGG16')

    model.summary()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def loadFigshareData(filePath):
    data_dir = f'{filePath}/Brain_MRI2/BRAIN_DATA'
    total_image = 3064

    trainindata = []
    for i in range(1, total_image + 1):
      filename = str(i) + ".mat"
      data = h5py.File(os.path.join(data_dir, filename), "r")
      trainindata.append(data)

      if i % 100 == 0:
        print(filename)

    random.shuffle(trainindata)

    # Now take all the image as train and test
    X = []
    y = []

    # For trainx and trainy
    for i in range(total_image):
      image = trainindata[i]["cjdata"]["image"][()]
      if image.shape == (512, 512):
        #image = np.expand_dims(image, axis=0)
        #rgb_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        X.append(image)
        # [()] operation is used to extract the value of the object
        # [0][0] is needed at the end because it is a 2 dimension array with one value and we have to take out the scalar from it
        label = int(trainindata[i]["cjdata"]["label"][()][0][0]) - 1
        y.append(label)

    del trainindata
    gc.collect()

    # Converting list to numpy array\
    print(f"image shape {image.shape}")
    X = np.array(X)
    X = np.dstack([X] * 3)
    X = X.reshape(-1, 512, 512, 3)

    #X = np.array(X).reshape(-1, 512, 512, 3)
    y = np.array(y)

    print(X.shape)
    print(y.shape)

    return X, y

def build_simple_cnn_figshare(input_shape):


    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((2, 2))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1))(X)
    print(X.shape)
    X = BatchNormalization(axis = 3)(X)
    print(X.shape)
    X = Activation('relu')(X) # shape=(?, 238, 238, 32)

    # MAXPOOL
    X = MaxPooling2D((4, 4))(X) # shape=(?, 59, 59, 32)

    # MAXPOOL
    X = MaxPooling2D((4, 4))(X) # shape=(?, 14, 14, 32)
    print(X.shape)
    # FLATTEN X
    X = Flatten()(X) # shape=(?, 6272)
    print(X.shape)
    # FULLYCONNECTED
    X = Dense(3, activation='softmax')(X) # shape=(?, 1)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='SimpleCNN')

    return model