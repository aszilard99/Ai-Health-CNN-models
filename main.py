import gc

import tensorflow as tf
import os
import numpy
import matplotlib.pyplot as plt
import h5py
import cv2
from random import randint
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time

from kaggle_brain_utils import  crop_brain_contour, load_image, load_data, build_vgg16extended_model, build_vgg16_model, build_simple_cnn, hms_string, split_data, plot_metrics, measureModelPerformance, measureModelPerformanceMulticlass
from figshare_dataset_utils import loadFigshareData, build_vgg16extended_model_figshare, build_vgg16_model_figshare, build_simple_cnn_figshare


#os.environ["TF_DIRECTML_MAX_ALLOC_SIZE"] = "536870912" # 512MB
os.environ["TF_DIRECTML_MAX_ALLOC_SIZE"] = "1536870912" # 1.5GB

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

def runSavedModelWithStatistics() :
    model = createModel()
    savedModel = loadModelWeights(model)
    trainx, testx, trainy, testy = loadFigshareData()
    measureModelPerformance(model=model, testx=testx, testy=testy)


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

    model = build_vgg16extended_model(IMG_SHAPE)
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

    model.fit(x=X_train, y=y_train, batch_size=32, epochs=40, validation_data=(X_val, y_val),
              callbacks=[tensorboard, checkpoint])

    end_time = time.time()
    execution_time = (end_time - start_time)
    print(f"Elapsed time: {hms_string(execution_time)}")

    history = model.history.history

    plot_metrics(history)

    measureModelPerformance(model=model, testx=X_test, testy=y_test)

def vgg16WithKaggleBrain():
    IMG_WIDTH, IMG_HEIGHT = (224, 224)
    IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)

    X, y = load_data(filePath + '/' + 'augmented_data', ['no', 'yes'], (IMG_WIDTH, IMG_HEIGHT))
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)

    model = build_vgg16_model(IMG_SHAPE)
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # tensorboard
    log_file_name = f'brain_tumor_detection_cnn_{int(time.time())}'
    tensorboard = TensorBoard(log_dir=f'logs/{log_file_name}')

    # checkpoint
    # unique file name that will include the epoch and the validation (development) accuracy
    modelPath = f"{filePath}/vgg16-cnn-parameters-improvement.model"
    # save the model with the best validation (development) accuracy till now
    checkpoint = ModelCheckpoint(modelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    start_time = time.time()

    model.fit(x=X_train, y=y_train, batch_size=32, epochs=40, validation_data=(X_val, y_val),
              callbacks=[tensorboard, checkpoint])

    end_time = time.time()
    execution_time = (end_time - start_time)
    print(f"Elapsed time: {hms_string(execution_time)}")

    history = model.history.history

    plot_metrics(history)

    measureModelPerformance(model=model, testx=X_test, testy=y_test)

def simpleCnnWithKaggleBrain():
    IMG_WIDTH, IMG_HEIGHT = (240, 240)
    IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)

    X, y = load_data(filePath + '/' + 'augmented_data', ['no', 'yes'], (IMG_WIDTH, IMG_HEIGHT))
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)

    model = build_simple_cnn(IMG_SHAPE)
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # tensorboard
    log_file_name = f'brain_tumor_detection_cnn_{int(time.time())}'
    tensorboard = TensorBoard(log_dir=f'logs/{log_file_name}')

    # checkpoint
    # unique file name that will include the epoch and the validation (development) accuracy
    modelPath = f"{filePath}/simple-cnn-kaggle_dataset.pb"
    # save the model with the best validation (development) accuracy till now
    checkpoint = ModelCheckpoint(modelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    start_time = time.time()

    model.fit(x=X_train, y=y_train, batch_size=32, epochs=40, validation_data=(X_val, y_val),
              callbacks=[tensorboard, checkpoint])

    end_time = time.time()
    execution_time = (end_time - start_time)
    print(f"Elapsed time: {hms_string(execution_time)}")

    history = model.history.history
    plot_metrics(history)

    # delete the the model the fit method returned, and load the best one that was saved during training
    del model

    model = load_model(f"{filePath}/simple-cnn-kaggle_dataset.pb")
    measureModelPerformance(model=model, testx=X_test, testy=y_test)

def simpleCNNPredict():

    model = load_model(f"{filePath}/simple-cnn-kaggle-brain.model")

    path = filePath + '/' + 'augmented_data' + '/' + 'yes' + '/' + 'aug_Y1_0_2900.jpg'

    image = load_image(path, (240, 240))

    X = np.array(image)
    X = np.expand_dims(X, 0)

    t0 = time.clock();
    pred = model.predict(X)
    t1 = time.clock();

    print(f'time {t1 - t0}')
    print(f'pred {pred}')


def vgg16ExtendedWithFigshareDataset():
    IMG_SHAPE = (512, 512, 3)

    X, y = loadFigshareData(filePath=filePath)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)

    print(len(X_train))
    print(len(y_train))
    print(len(X_val))
    print(len(y_val))
    print(len(X_test))
    print(len(y_test))

    model = build_vgg16extended_model_figshare(IMG_SHAPE)

    log_file_name = f'vgg16_extended_with_figshare_dataset_{int(time.time())}'
    tensorboard = TensorBoard(log_dir=f'logs/{log_file_name}')

    # checkpoint
    # unique file name that will include the epoch and the validation (development) accuracy
    modelPath = f"{filePath}/vgg16_extended_with_figshare_dataset.model"
    # save the model with the best validation (development) accuracy till now
    checkpoint = ModelCheckpoint(modelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    start_time = time.time()

    model.fit(x=X_train, y=y_train, batch_size=32, epochs=80, validation_data=(X_val, y_val),
              callbacks=[tensorboard, checkpoint])

    end_time = time.time()
    execution_time = (end_time - start_time)
    print(f"Elapsed time: {hms_string(execution_time)}")

    history = model.history.history

    plot_metrics(history)

    measureModelPerformanceMulticlass(model=model, testx=X_test, testy=y_test)

def vgg16WithFigshareDataset():
    IMG_SHAPE = (512, 512, 3)

    X, y = loadFigshareData(filePath=filePath)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)

    print(len(X_train))
    print(len(y_train))
    print(len(X_val))
    print(len(y_val))
    print(len(X_test))
    print(len(y_test))

    model = build_vgg16_model_figshare(IMG_SHAPE)

    #log_file_name = f'vgg16_with_figshare_dataset_{int(time.time())}'
    #tensorboard = TensorBoard(log_dir=f'logs/{log_file_name}')

    # checkpoint
    # unique file name that will include the epoch and the validation (development) accuracy
    #modelPath = f"{filePath}/vgg16_with_figshare_dataset.model"
    # save the model with the best validation (development) accuracy till now
    #checkpoint = ModelCheckpoint(modelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    start_time = time.time()

    model.fit(x=X_train, y=y_train, batch_size=32, epochs=2, validation_data=(X_val, y_val))

    end_time = time.time()
    execution_time = (end_time - start_time)
    print(f"Elapsed time: {hms_string(execution_time)}")

    history = model.history.history

    plot_metrics(history)

    measureModelPerformanceMulticlass(model=model, testx=X_test, testy=y_test)

def simpleCnnWithFigshareDataset():
    IMG_SHAPE = (512, 512, 3)

    X, y = loadFigshareData(filePath=filePath)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)

    print(f"X_train.shape {X_train.shape}")
    print(f"X_val.shape {X_val.shape}")
    print(f"X_test.shape {X_test.shape}")


    model = build_simple_cnn_figshare(IMG_SHAPE)

    model.summary()

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # tensorboard
    log_file_name = f'simple-cnn-fighshare-dataset_{int(time.time())}'
    tensorboard = TensorBoard(log_dir=f'logs/{log_file_name}')

    # checkpoint
    # unique file name that will include the epoch and the validation (development) accuracy
    modelPath = f"{filePath}/simple-cnn-fighshare-dataset2.model"
    # save the model with the best validation (development) accuracy till now
    checkpoint = ModelCheckpoint(modelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    start_time = time.time()

    model.fit(x=X_train, y=y_train, batch_size=32, epochs=40, validation_data=(X_val, y_val),
              callbacks=[tensorboard, checkpoint])

    end_time = time.time()
    execution_time = (end_time - start_time)
    print(f"Elapsed time: {hms_string(execution_time)}")

    history = model.history.history
    plot_metrics(history)

    #delete the the model the fit method returned, and load the best one that was saved during training
    del model
    model = load_model(f"{filePath}/simple-cnn-fighshare-dataset2.model")
    measureModelPerformanceMulticlass(model=model, testx=X_test, testy=y_test)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    simpleCnnWithKaggleBrain()

    #simpleCNNPredict()

    #trainx, testx, trainy, testy = loadData()
    #model = build_model_vgg16_plus();
    #trainModel(model, trainx, testx, trainy, testy)

    #displayImage(12);
    #plot_model(model, to_file='vgg16_plus_nn.png', show_shapes=True, show_layer_names=True)


