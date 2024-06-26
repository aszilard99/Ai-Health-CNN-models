from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D,Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
import seaborn as sns
import pandas as pd
import gc
from os import listdir

def load_image(path, image_size):

    image_width, image_height = image_size

    # load the image
    image = plt.imread(path)
    #this is done by the process_image function

    # # crop the brain and ignore the unnecessary rest part of the image
    # image = crop_brain_contour(image, plot=False)
    # # resize image
    # print("Resizing image")
    # image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
    # print("Resized image")
    # normalize values
    #image = image / 255.

    return image

def load_data(resources_path , dir_list, image_size):
    """
    Read images, resize and normalize them.
    Arguments:
        dir_list: list of strings representing file directories.
    Returns:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
    """

    # load all images in a directory
    X = []
    y = []
    counter = 0

    for directory in dir_list:
        print(resources_path + '/' + directory)
        for filename in listdir(resources_path + '/' + directory):
            print("loading images")
            image = load_image(resources_path + '/' + directory + '/' + filename, image_size)
            print(f"image shape {image.shape}, filename {filename}, counter {counter}")
            # convert image to numpy array and append it to X
            if counter == 0:
                X = image[np.newaxis, :, :, :]
            else:
                #a = np.append(a, c[np.newaxis, :], axis=0)
                X = np.append(X, image[np.newaxis, :, :, :], axis=0)
            # append a value of 1 to the target array if the image
            # is in the folder named 'yes', otherwise append 0.
            counter = counter + 1
            if directory[-3:] == 'yes':
                y.append([1])
            else:
                y.append([0])
    print("finished loading images")
    #X = np.array(X)
    y = np.array(y)
    print("finished converting images to np.array")
    # Shuffle the data
    #X, y = shuffle(X, y)
    #print("finished shuffleing images")

    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')

    return X, y

def process_batch_of_images(resources_path , dir_list, image_size):
    """
    Read images, resize and normalize and then save them.
    Arguments:
        dir_list: list of strings representing file directories.
        image_size: the size that the images should be resized to
    Returns:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
    """
    for directory in dir_list:
        for filename in listdir(resources_path + 'augmented_data' + '/' + directory):
            print("loading image")
            image = process_image(resources_path + 'augmented_data' + '/' + directory + '/' + filename, image_size)
            print("saving image to ")
            print(resources_path + 'resized_images' + '/' + directory + '/' + filename)
            cv2.imwrite(resources_path + 'resized_images' + '/' + directory + '/' + filename, image)
            print("saved image")

def process_image(path, image_size):

    image_width, image_height = image_size

    # load the image
    image = cv2.imread(path)

    # crop the brain and ignore the unnecessary rest part of the image
    image = crop_brain_contour(image, plot=False)
    # resize image
    print("Resizing image")
    image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
    print("Resized image")

    image = image / 255.

    return image

def crop_brain_contour(image, plot=False):

    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)


    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    if plot:
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.imshow(image)

        plt.tick_params(axis='both', which='both',
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(new_image)

        plt.tick_params(axis='both', which='both',
                        top=False, bottom=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        plt.title('Cropped Image')

        plt.show()

    return new_image

def plot_sample_images(X, y, n=50):
    """
    Plots n sample images for both values of y (labels).
    Arguments:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
    """

    for label in [0,1]:
        # grab the first n images with the corresponding y values equal to label
        images = X[np.argwhere(y == label)]
        n_images = images[:n]

        columns_n = 10
        rows_n = int(n/ columns_n)

        plt.figure(figsize=(20, 10))

        i = 1 # current plot
        for image in n_images:
            plt.subplot(rows_n, columns_n, i)
            plt.imshow(image[0])

            # remove ticks
            plt.tick_params(axis='both', which='both',
                            top=False, bottom=False, left=False, right=False,
                           labelbottom=False, labeltop=False, labelleft=False, labelright=False)

            i += 1

        label_to_str = lambda label: "Yes" if label == 1 else "No"
        plt.suptitle(f"Brain Tumor: {label_to_str(label)}")
        plt.show()

def split_data(X, y, test_size=0.2):

    """
    Splits data into training, development and test sets.
    Arguments:
        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)
        y: A numpy array with shape = (#_examples, 1)
    Returns:
        X_train: A numpy array with shape = (#_train_examples, image_width, image_height, #_channels)
        y_train: A numpy array with shape = (#_train_examples, 1)
        X_val: A numpy array with shape = (#_val_examples, image_width, image_height, #_channels)
        y_val: A numpy array with shape = (#_val_examples, 1)
        X_test: A numpy array with shape = (#_test_examples, image_width, image_height, #_channels)
        y_test: A numpy array with shape = (#_test_examples, 1)
    """

    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)

    del X
    del y
    gc.collect()

    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)

    del X_test_val
    del y_test_val
    gc.collect()

    return X_train, y_train, X_val, y_val, X_test, y_test

def compute_f1_score(y_true, prob):
    # convert the vector of probabilities to a target vector
    y_pred = np.where(prob > 0.5, 1, 0)

    score = f1_score(y_true, y_pred)

    return score

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{round(s,1)}"

def build_vgg16extended_model(input_shape):

  conv = VGG16(input_shape= input_shape, weights='imagenet',include_top=False)

  for layer in conv.layers:
    layer.trainable = False

  x = conv.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(1024,activation='relu')(x)
  x = Dense(1024,activation='relu')(x)
  x = Dense(512, activation='relu')(x)
  x = Dropout(.2)(x)
  pred = Dense(1,activation='sigmoid')(x)
  model = Model(inputs = conv.input,outputs=pred, name='VGG16Extended')

  return model

def build_vgg16_model(input_shape):

    conv = VGG16(input_shape=input_shape, weights='imagenet', include_top=False)

    for layer in conv.layers:
        layer.trainable = False

    x = conv.output
    x = Flatten()(x)
    x = Dense(units=4096,activation="relu")(x)
    x = Dense(units=4096,activation="relu")(x)
    pred = Dense(units=1, activation="sigmoid")(x)
    model = Model(inputs=conv.input, outputs=pred, name='VGG16')

    return model

def build_complex_cnn_with_kaggle_brain() :

    tf.keras.backend.clear_session()

    # Initial  BLock of the model
    ini_input = Input(shape=(512, 512, 3), name="image")

    x1 = Conv2D(64, (22, 22), strides=2, padding="same")(ini_input)
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
    pred = Dense(units=1, activation="sigmoid")(x8)

    model = Model(inputs=ini_input, outputs=pred)

    return model


def build_simple_cnn(input_shape):

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
    model = Model(inputs = X_input, outputs = X, name='SimpleCNN')

    return model

def plot_metrics(history):

    train_loss = history['loss']
    val_loss = history['val_loss']
    train_acc = history['acc']
    val_acc = history['val_acc']

    # Loss
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xticks(np.arange(0, 44, 4))
    plt.ylabel('Loss Value')
    plt.xlabel('Epoch')
    plt.title('Loss')
    plt.legend()
    plt.show()

    # Accuracy
    plt.figure()
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xticks(np.arange(0, 44, 4))
    plt.ylabel('Accuracy Value')
    plt.xlabel('Epoch')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

def measureModelPerformance(model, testx, testy):

    pred = model.predict(testx)

    predThreshold = list(map(threshold, pred))

    target_names = ['No tumor', 'Tumor']

    print(f'{pred.shape} pred.shape')

    print(f'{testy.shape} testy.shape')

    tn, fp, fn, tp = confusion_matrix(testy, predThreshold).ravel()
    print(f"tn:{tn} fp:{fp} fn:{fn} tp:{tp}")

    print('Confusion Matrix')
    print(confusion_matrix(testy, predThreshold))

    print('Classification Report')
    print(classification_report(testy, predThreshold, target_names=target_names))

    cm = confusion_matrix(testy, predThreshold)

    df_cm = pd.DataFrame(cm, target_names, target_names)
    sns.heatmap(df_cm, annot=True, cmap='viridis', fmt='d')
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.title('Confusion Matrix')
    plt.show()

    ax = plt.subplot()
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(target_names)
    ax.yaxis.set_ticklabels(target_names)

    fpr, tpr, thresholds = metrics.roc_curve(testy, pred)
    metrics.auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fpr, tpr, label='actual', linestyle='solid')
    ax.plot(np.linspace(0, 1, 100),
            np.linspace(0, 1, 100),
            label='baseline',
            linestyle='--')
    plt.title('Receiver Operating Characteristic Curve', fontsize=14)
    plt.ylabel('Total Positive Rate', fontsize=12)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.legend(fontsize=12)
    plt.show()

def measureModelPerformanceMulticlass(model, testx, testy) :

    pred = model.predict(testx)

    Y_pred = np.argmax(pred, 1)

    target_names = ['Meningioma', 'Glioma', 'Pituitary']

    print(f'{Y_pred.shape} pred.shape')
    print(f'{Y_pred} pred')

    print(f'{testy.shape} testy.shape')
    print(f'{testy} testy')

    print('Confusion Matrix')
    print(confusion_matrix(testy, Y_pred))

    print('Classification Report')
    print(classification_report(testy, Y_pred, target_names=target_names, labels=np.arange(0,len(target_names),1)))

    cm = confusion_matrix(testy, Y_pred)

    df_cm = pd.DataFrame(cm, target_names, target_names)
    sns.heatmap(df_cm, annot=True, cmap='viridis', fmt='d')
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.title('Confusion Matrix')
    plt.show()

    ax = plt.subplot()
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(target_names)
    ax.yaxis.set_ticklabels(target_names)

def threshold(n):
    return 1 if n >= 0.5 else 0

def getMax(list):
    print(f"max(list) {max(list)}")
    return max(list)