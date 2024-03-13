import tensorflow as tf
import os
import numpy
import matplotlib.pyplot as plt
import h5py
import cv2
from random import randint
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import time
from tensorflow.keras.utils import plot_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py

from kaggle_brain_utils import  crop_brain_contour, load_data, build_model, hms_string, split_data, plot_metrics, createStatistics

epochs = 2
basePath = Path(__file__).parent
filePath = (f"{basePath}/resources")

def loadModelWeights(model) :

    savedModel = model.load_weights(f'{filePath}/my_model_80epoch.h5')
    return savedModel

def trainModel(model, trainx, testx, trainy, testy) :
    r = model.fit(trainx,
                  trainy,
                  epochs=epochs,
                  batch_size=32,
                  verbose=1,
                  validation_data=(testx, testy),
                  shuffle=False
                  )

    num = randint(0,1000000)
    model.save(f'{filePath}/my_model_epochs_{epochs}_{num}.h5')

    model.save_weights(f'{filePath}/my_weight_epochs_{epochs}__{num}')

def loadData() :
    data_dir = f'{filePath}/Brain_MRI2/BRAIN_DATA'
    total_image = 3064
    trainindata = []
    for i in range(1, total_image + 1):
        filename = str(i) + ".mat"
        data = h5py.File(os.path.join(data_dir, filename), "r")
        trainindata.append(data)

        if i % 100 == 0:
            print(filename)

    print(trainindata[5])

    import keras
    import random

    random.shuffle(trainindata)

    import numpy as np

    # Now take all the image as train and test
    trainx = []
    trainy = []
    testx = []
    testy = []

    size = round(4 * total_image / 5)  # Split the dataset into 80:20
    # For trainx and trainy
    for i in range(size):
        image = trainindata[i]["cjdata"]["image"][()]
        if image.shape == (512, 512):
            image = np.expand_dims(image, axis=0)
            trainx.append(image)

            # [()] operation is used to extract the value of the object
            # [0][0] is needed at the end because it is a 2 dimension array with one value and we have to take out the scalar from it
            label = int(trainindata[i]["cjdata"]["label"][()][0][0]) - 1
            trainy.append(label)
    # For trainx and trainy
    for i in range(size, total_image):
        image = trainindata[i]["cjdata"]["image"][()]
        if image.shape == (512, 512):
            image = np.expand_dims(image, axis=0)
            testx.append(image)
            label = int(trainindata[i]["cjdata"]["label"][()][0][0]) - 1
            testy.append(label)

    # Converting list to numpy array
    trainx = np.array(trainx).reshape(-1, 512, 512, 1)
    testx = np.array(testx).reshape(-1, 512, 512, 1)
    trainy = np.array(trainy)
    testy = np.array(testy)

    print(trainx.shape)
    print(testx.shape)
    print(trainy.shape)
    print(testy.shape)

    return trainx, testx, trainy, testy

def createModel() :
    # Model building starts
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
    from tensorflow.keras.layers import BatchNormalization
    import numpy as np

    tf.keras.backend.clear_session()

    # Initial  BLock of the model
    ini_input = keras.Input(shape=(512, 512, 1), name="image")

    x1 = layers.Conv2D(64, (22, 22), strides=2)(ini_input)
    x1 = layers.MaxPooling2D((4, 4))(x1)
    x1 = layers.BatchNormalization()(x1)

    x2 = layers.Conv2D(128, (11, 11), strides=2, padding="same")(x1)
    x2 = layers.MaxPooling2D((2, 2))(x2)
    x2 = layers.BatchNormalization()(x2)

    x3 = layers.Conv2D(256, (7, 7), strides=2, padding="same")(x2)
    x3 = layers.MaxPooling2D((2, 2))(x3)
    x3 = layers.BatchNormalization()(x3)

    x4 = layers.Conv2D(512, (3, 3), strides=2, padding="same")(x3)
    x4 = layers.MaxPooling2D((2, 2))(x4)
    x4 = layers.BatchNormalization()(x4)

    x5 = layers.GlobalAveragePooling2D()(x4)
    x5 = layers.Activation("relu")(x5)

    x6 = layers.Dense(1024, "relu")(x5)
    x6 = layers.BatchNormalization()(x6)
    x7 = layers.Dense(512, "relu")(x6)
    x7 = layers.BatchNormalization()(x7)
    x8 = layers.Dense(256, "relu")(x7)
    x8 = layers.BatchNormalization()(x8)
    x8 = layers.Dropout(.2)(x8)
    x9 = layers.Dense(3)(x8)
    pred = layers.Activation("softmax")(x9)

    model = keras.Model(inputs=ini_input, outputs=pred)

    model.summary()

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

def build_model_seven_layer():

    import tensorflow as tf
    from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, \
        Flatten, Dense
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from sklearn.utils import shuffle
    #import cv2
    import imutils
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    from os import listdir

    input_shape = (240, 240, 3);

    """
    Arugments:
        input_shape: A tuple representing the shape of the input of the model. shape=(image_width, image_height, #_channels)
    Returns:
        model: A Model object.
    """
    # Define the input placeholder as a tensor with shape input_shape.
    X_input = Input(input_shape) # shape=(?, 240, 240, 3)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((2, 2))(X_input) # shape=(?, 244, 244, 3)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X) # shape=(?, 238, 238, 32)

    # MAXPOOL
    X = MaxPooling2D((4, 4))(X) # shape=(?, 59, 59, 32)

    # MAXPOOL
    X = MaxPooling2D((4, 4))(X) # shape=(?, 14, 14, 32)

    # FLATTEN X
    X = Flatten()(X) # shape=(?, 6272)
    # FULLYCONNECTED
    X = Dense(1, activation='sigmoid')(X) # shape=(?, 1)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='BrainDetectionModel')

    return model

def build_model_vgg16_plus():
    import tensorflow as tf
    from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, \
        Flatten, Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
    from tensorflow.keras.applications.vgg16 import VGG16
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from sklearn.utils import shuffle
    #import cv2
    import imutils
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    from os import listdir

    input_shape = (240, 240, 3)

    conv = VGG16(input_shape= input_shape, weights='imagenet',include_top=False)

    for layer in conv.layers:
        layer.trainable = False

    x = conv.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    x = Dense(1024,activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(.2)(x)
    pred = Dense(2, activation='sigmoid')(x)
    model = Model(inputs = conv.input, outputs=pred, name='VGG16PlusBrainDetectionModel')

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

def runSavedModelWithStatistics() :
    model = createModel()
    savedModel = loadModelWeights(model)
    trainx, testx, trainy, testy = loadData()
    createStatistics(model=model, testx=testx, testy=testy)


def displayImage(fileName) :

    with h5py.File(f'{filePath}/Brain_MRI2/BRAIN_DATA/{fileName}.mat', 'r') as file:
        data = file.get('cjdata/image')
        print(data)
        data = np.array(data)
        imgplot = plt.imshow(data)
        plt.show()

def exploratoryDataAnalysis():

    data_dir = f'{filePath}/Brain_MRI2/BRAIN_DATA'

    with h5py.File(f'{data_dir}/2204.mat', 'r') as file:
        print(file['cjdata'].keys())
        print(file['cjdata']['label'][()][0][0])
        print(file['cjdata']['tumorMask'])
        print(file['cjdata']['tumorBorder'])

    total_image = 3064
    trainindata = []
    labelsCountTable = [0, 0, 0]
    labels = []
    for i in range(1, total_image + 1):
        filename = str(i) + ".mat"
        data = h5py.File(os.path.join(data_dir, filename), "r")

        trainindata.append(data)

        label = int(data["cjdata"]["label"][()][0][0]) - 1
        labels.append(label)
        labelsCountTable[label] = labelsCountTable[label] + 1

        if i % 100 == 0:
            print(filename)

    print(trainindata[3011].keys())
    print(trainindata[3011]['cjdata'].keys())
    print(f"label {trainindata[3011]['cjdata']['label'][()][0][0]}")
    print(trainindata[3011]['cjdata']['tumorMask'])
    print(trainindata[3011]['cjdata']['tumorBorder'])

    print(f'labels count table {labelsCountTable}')

    target_labels = ['Meningioma', 'Glioma', 'Pituitary']

    x = np.array(target_labels)
    y = np.array(labelsCountTable)

    plt.bar(x, y)
    plt.title('Number of images for each tumor type')
    plt.show()


def vgg16ExtendedWithKaggleBrain():
    IMG_WIDTH, IMG_HEIGHT = (240, 240)
    IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)

    X, y = load_data(filePath+'/' + 'augmented_data',['no', 'yes'], (IMG_WIDTH, IMG_HEIGHT))
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)

    model = build_model(IMG_SHAPE)
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # tensorboard
    log_file_name = f'brain_tumor_detection_cnn_{int(time.time())}'
    tensorboard = TensorBoard(log_dir=f'logs/{log_file_name}')

    # checkpoint
    # unique file name that will include the epoch and the validation (development) accuracy
    modelPath = f"{filePath}/vgg16extended-cnn-parameters-improvement.model"
    # save the model with the best validation (development) accuracy till now
    checkpoint = ModelCheckpoint(modelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    start_time = time.time()

    model.fit(x=X_train, y=y_train, batch_size=32, epochs=4, validation_data=(X_val, y_val),
              callbacks=[tensorboard, checkpoint])

    end_time = time.time()
    execution_time = (end_time - start_time)
    print(f"Elapsed time: {hms_string(execution_time)}")

    history = model.history.history

    #plot_metrics(history)

    createStatistics(model=model, testx=X_test, testy=y_test)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    vgg16ExtendedWithKaggleBrain()

    #trainx, testx, trainy, testy = loadData()
    #model = build_model_vgg16_plus();
    #trainModel(model, trainx, testx, trainy, testy)

    #displayImage(12);
    #plot_model(model, to_file='vgg16_plus_nn.png', show_shapes=True, show_layer_names=True)


