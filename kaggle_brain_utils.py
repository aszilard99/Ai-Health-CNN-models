from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D,Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
import seaborn as sns

from os import listdir

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
    image_width, image_height = image_size

    for directory in dir_list:
        for filename in listdir(resources_path + '/' + directory):
            # load the image
            image = cv2.imread(resources_path + '/' + directory + '/' + filename)
            # crop the brain and ignore the unnecessary rest part of the image
            image = crop_brain_contour(image, plot=False)
            # resize image
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
            # normalize values
            image = image / 255.
            # convert image to numpy array and append it to X
            X.append(image)
            # append a value of 1 to the target array if the image
            # is in the folder named 'yes', otherwise append 0.
            if directory[-3:] == 'yes':
                y.append([1])
            else:
                y.append([0])

    X = np.array(X)
    y = np.array(y)

    # Shuffle the data
    X, y = shuffle(X, y)

    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')

    return X, y
def crop_brain_contour(image, plot=False):

    #import imutils
    #import cv2
    #from matplotlib import pyplot as plt

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
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)

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

def build_model(input_shape):

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
  model = Model(inputs = conv.input,outputs=pred, name='VGG16PlusBrainDetectionModel')

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
    plt.title('Loss')
    plt.legend()
    plt.show()

    # Accuracy
    plt.figure()
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

def createStatistics(model, testx, testy) :

    pred = model.predict(testx)

    predThreshold = list(map(threshold, pred))

    print(f'{pred.shape} pred.shape')

    print(f'{testy.shape} testy.shape')

    #print(f'pred {pred}')
    #print(f'Y_pred {Y_pred}')
    #print(f'testy {testy}')
    #print(f'predThreshold {predThreshold}')
    print('Confusion Matrix')
    print(confusion_matrix(testy, predThreshold))

    cm = confusion_matrix(testy, predThreshold)
    sns.heatmap(cm, square=True, annot=True, cbar=False, cmap=plt.cm.Blues, fmt='d')
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')

    print('Classification Report')
    target_names = ['Tumor', 'No tumor']
    print(classification_report(testy, predThreshold, target_names=target_names))

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='d')  # annot=True to annotate cells
    plt.show()

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(target_names)
    ax.yaxis.set_ticklabels(target_names)

    results = confusion_matrix(testy, predThreshold)
    print('Confusion Matrix :')
    print(results)
    print('Accuracy Score :', accuracy_score(testy, predThreshold))
    print('Report : ')
    print(classification_report(testy, predThreshold))

    sns.heatmap(results / np.sum(results), annot=True,
                fmt='.2%', cmap='Blues')

    fpr, tpr, thresholds = metrics.roc_curve(testy, predThreshold, pos_label=2)
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
def threshold(n):
     return 1 if n >= 0.5 else 0