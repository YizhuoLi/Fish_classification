# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from skimage import io
from skimage.transform import resize
from skimage import color
from skimage import feature



# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
IMAGE_DIMS = (96, 96, 3)

ss = StandardScaler()
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
    help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
    help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
    help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the data and labels
data = []
labels = []

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    greyscale_image = color.rgb2gray(image) # Restricting the dimension of our data from 3D to 2D    
    # Extract features (skimage features)
    featureVector = feature.hog(greyscale_image, orientations=9, pixels_per_cell=(8,8), cells_per_block=(1,1))
    data.append(featureVector)


    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(int(label.replace('fish_','')))

# print(labels)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0 
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
    data.nbytes / (1024 * 1000.0)))

print(labels)
# data = data.flatten()   

# binarize the labels
# lb = LabelBinarizer()
# labels = lb.fit_transform(labels)

# x = np.asarray(data)
# y = np.asarray(labels)

# print(x.shape)
# print(y.shape)


# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(X_train, X_test, y_train, y_test) = train_test_split(data,
    labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# initialize the model
print("[INFO] compiling model...")        
svm = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

# Train the model
print("[INFO] Trainning model....")
svm.fit(X_train, y_train)
test_score = svm.score(X_test, y_test)

# print("Best score on cross-validation: {:0.2f}".format(best_score))
# print("Best parameters: {}".format(best_param))
print("Test set score: {:.2f}".format(test_score))