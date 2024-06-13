from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D,Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16
import h5py
import os
import random
import numpy as np
import gc

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

def loadFigshareData(filePath, isImageSingleColorChannel):
    data_dir = f'{filePath}/Brain_MRI2/BRAIN_DATA'
    total_image = 3064

    trainindata = []
    for i in range(1, total_image + 1):
      filename = str(i) + ".mat"
      data = h5py.File(os.path.join(data_dir, filename), "r")
      trainindata.append(data)

      if i % 100 == 0:
        print(filename)
    print("shuffling")
    random.shuffle(trainindata)
    print("shuffled")
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

    print("reshaping data")
    X, y = reshapeData(X, y, isImageSingleColorChannel)
    print("reshaped data")

    return X, y

def reshapeData(X, y, isImageSingleColorChannel):
    if (isImageSingleColorChannel == True):
        print(f"single color channel mode")
        X = np.array(X)
        X = X.reshape(-1, 512, 512, 1)

        y = np.array(y)

        print(X.shape)
        print(y.shape)
        return X, y
    else:
        print(f"multi color channel mode")
        X = np.array(X)
        X = np.dstack([X] * 3)
        X = X.reshape(-1, 512, 512, 3)

        y = np.array(y)

        print(X.shape)
        print(y.shape)
        return X,y

def build_complex_cnn_figshare():

    # Initial  BLock of the model
    ini_input = Input(shape=(512, 512, 1), name="image")

    x1 = Conv2D(64, (22, 22), strides=2)(ini_input)
    x1 = MaxPooling2D((4, 4))(x1)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(128, (11, 11), strides=2, padding="same")(x1)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = BatchNormalization()(x2)

    x3 = Conv2D(256, (7, 7), strides=2, padding="same")(x2)
    x3 = MaxPooling2D((2, 2))(x3)
    x3 = BatchNormalization()(x3)

    x4 = Conv2D(512, (3, 3), strides=2, padding="same")(x3)
    x4 = MaxPooling2D((2, 2))(x4)
    x4 = BatchNormalization()(x4)

    x5 = GlobalAveragePooling2D()(x4)
    x5 = Activation("relu")(x5)

    x6 = Dense(1024, "relu")(x5)
    x6 = BatchNormalization()(x6)
    x7 = Dense(512, "relu")(x6)
    x7 = BatchNormalization()(x7)
    x8 = Dense(256, "relu")(x7)
    x8 = BatchNormalization()(x8)
    x8 = Dropout(.2)(x8)
    pred = Dense(units=3, activation="softmax")(x8)

    model = Model(inputs=ini_input, outputs=pred)

    return model

def build_simple_cnn_kaggle_brain(input_shape):


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