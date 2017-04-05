print "myKeras_FaceNet_NN1.py"

# images is an ndarray of min. 2 dimensions
def resizeImages(images, imRows, imCols):

    print "resizing..."

    # image : rows x cols
    if (images.ndim==2):
        newImage = np.zeros((imRows, imCols))
        newImage = imresize(images, (imRows, imCols))/255.0
        return newImage

    # image : rows x cols x channels
    elif (images.ndim==3):
        newImages = np.zeros((imRows, imCols, images.shape[2]))
        for ch in range(images.shape[2]):
            newImage[:,:,ch] = imresize(images[:,:,ch], (imRows, imCols))/255.0

        print "ndims 3"
        return newImages

    # image : nOfImages x rows x cols x channels
    elif (images.ndim==4):
        newImages = np.zeros((images.shape[0], imRows, imCols, images.shape[3]))
        for i in range(images.shape[0]):
            for ch in range(images.shape[3]):
                newImages[i,:,:,ch] = imresize(images[i,:,:,ch], (imRows, imCols))/255.0

        return newImages

    else:
        print "image dimensions not 2, 3 or 4"
        return False
 

'''
# Paths
basePath = os.getcwd()
modelsPath = basePath + '/../MODELS'

# import stuff
sys.path.append(modelsPath)
from importStuffLoadData import *

# for plotting
import matplotlib.pyplot as plt
from pylab import text

nbClasses = 1
date = '2016_11_03'

# for entropies
nbBins = 12

[origImRows, origImCols, origImChannels, (trainImages, trainResults), (valImages, valResults), (testImages, testResults), fullTestImages] = loadData(nbClasses, date)

os.chdir(basePath)


newTrainImages = resizeImages(trainImages, imRows, imCols)
newValImages = resizeImages(valImages, imRows, imCols)
newTestImages = resizeImages(testImages, imRows, imCols)

'''

import numpy as np

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

import os
import sys
import time

# keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Lambda, MaxoutDense
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.normalization import LRN2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from sklearn.metrics import mean_squared_error

# Save time
timeStr = time.strftime('%Y%m%d_%H%M%S')

# imresize
imRows = 220
imCols = 220
imChannels = 3
newTrainImages = resizeImages(trainImages, imRows, imCols)
newValImages = resizeImages(valImages, imRows, imCols)

# create model

batchSize = 1800
nbEpochs = 100

nn1 = Sequential() #220x220x3

# conv1
nn1.add(Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu', input_shape=(imRows, imCols, imChannels)))

# pool1
nn1.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same'))

# rnorm1
nn1.add(LRN2D())

# conv2a
nn1.add(Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))

# conv2
nn1.add(Conv2D(192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

# rnorm2
nn1.add(LRN2D())

# pool2
nn1.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same'))

# conv3a
nn1.add(Conv2D(192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))

# conv3
nn1.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

# pool3
nn1.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same'))

# conv4a
nn1.add(Conv2D(384, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))

# conv4
nn1.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

# conv5a
nn1.add(Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))

# conv5
nn1.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

# conv6a
nn1.add(Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))

# conv6
nn1.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

# pool4
nn1.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same'))

# concat
nn1.add(Flatten())

# fc1
nn1.add(MaxoutDense(4096, nb_feature=2))

# fc2
nn1.add(MaxoutDense(4096, nb_feature=2))

# fc7128
nn1.add(Dense(128))

# L2
nn1.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))


# Compile the model
nn1.compile(optimizer='adam', loss='binary_crossentropy')

'''
# Fit the model
nn1.fit(newTrainImages, newTrainImages,
                batch_size=batchSize,
                nb_epoch=nbEpochs,
                shuffle=True,
                validation_data=(newValImages, newValImages),
                verbose=1)


# Save the model
# Save time
timeStr = time.strftime('%Y%m%d_%H%M%S')
modelName = 'myKerasSmallNNModel_autoencoder_2layers_' + timeStr + '.h5'
nn1.save(modelName)
print "model saved."

# Load the model
modelName = modelsPath + '/myKerasSmallNNModel_autoencoder_2layers_20161112_184710.h5'
autoencoder = load_model(modelName)
'''

