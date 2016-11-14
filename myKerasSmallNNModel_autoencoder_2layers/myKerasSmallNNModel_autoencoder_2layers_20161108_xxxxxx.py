print "myKerasSmallNN_autoencoder_2layersr.py"
basePath = '/home/gor/projects/vikram/myKerasNN/myKerasSmallNNModel_autoencoder_2layers'
modelsPath = '/home/gor/projects/vikram/myKerasNN/MODELS/'

import numpy as np

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

import os
import numpy as np
import matplotlib.pyplot as plt
from pylab import text
import sys
import time

from keras.models import Sequential, load_model
from keras.layers import Dense, Convolution2D, MaxPooling2D, UpSampling2D
from sklearn.metrics import mean_squared_error

sys.path.append(modelsPath)
from importStuffLoadData import *

# Save time
timeStr = time.strftime('%Y%m%d_%H%M%S')

nbClasses = 1
#date = '2016_08_19'
date = '2016_11_03'

# for entropies
nbBins = 12

[origImRows, origImCols, origImChannels, (trainImages, trainResults), (valImages, valResults), (testImages, testResults), fullTestImages] = loadData(nbClasses, date)

os.chdir(basePath)

# TODO: KL transform
# TODO: Covariance Equalization

# imresize
imRows = 64
imCols = 64
imChannels = 1

newTrainImages = resizeImages(trainImages, imRows, imCols)
newValImages = resizeImages(valImages, imRows, imCols)
newTestImages = resizeImages(testImages, imRows, imCols)

# create model
#Autoencoder
batchSize = 128
nbEpochs = 100

autoencoder = Sequential() #64x64x1
autoencoder.add(Convolution2D(16, 9, 9, input_shape=(imRows, imCols, imChannels), border_mode='same',  activation='relu')) #64x64x16
autoencoder.add(MaxPooling2D(pool_size=(2, 2))) #32x32x16
autoencoder.add(Convolution2D(8, 5, 5, border_mode='same', activation='relu')) #32x32x8
autoencoder.add(MaxPooling2D(pool_size=(2, 2))) #16x16x8

# at this point the representation is (16, 16, 8)

autoencoder.add(Convolution2D(8, 5, 5, border_mode='same', activation='relu')) #16x16x8
autoencoder.add(UpSampling2D(size=(2, 2))) #32x32x8
autoencoder.add(Convolution2D(16, 9, 9, border_mode='same', activation='relu')) #32x32x16
autoencoder.add(UpSampling2D(size=(2, 2))) #64x64x16
autoencoder.add(Convolution2D(1, 3, 3, border_mode='same', activation='sigmoid')) #64x64x1

# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Fit the model
autoencoder.fit(newTrainImages, newTrainImages,
                batch_size=batchSize,
                nb_epoch=nbEpochs,
                shuffle=True,
                validation_data=(newValImages, newValImages),
                verbose=1)


# Save the model
# Save time
timeStr = time.strftime('%Y%m%d_%H%M%S')
modelName = 'myKerasSmallNNModel_autoencoder_2layers_' + timeStr + '.h5'
autoencoder.save(modelName)
print "model saved."

'''
# Load the modela
'''

# FINDING RANGE OF ENTROPIES

# calc entropy for train
trainPreds = autoencoder.predict(newTrainImages)
trainEntropies = calcCrossEntropy(newTrainImages, trainPreds, nbBins)

'''
for i in range(trainPreds.shape[0]):
	plotImage = np.concatenate((newTrainImages[i].reshape(imRows, imCols), trainPreds[i].reshape(imRows, imCols)), axis=1)
	plt.imshow(plotImage, cmap='gray')
	plt.show()
'''

# predict, calc entropy for val
valPreds = autoencoder.predict(newValImages)
valEntropies = calcCrossEntropy(newValImages, valPreds, nbBins)

# predict, calc entropy for test
testPreds = autoencoder.predict(newTestImages)
testEntropies = calcCrossEntropy(testImages, testPreds, nbBins)


# Entropies

print "trainEntropies : " + trainEntropies.min().astype(str) + " ---- " + trainEntropies.max().astype(str)
print "valEntropies   : " + valEntropies.min().astype(str) + " ---- " + valEntropies.max().astype(str)
print "testEntropies  : " + testEntropies.min().astype(str) + " ---- " + testEntropies.max().astype(str)

minEntropy = np.array([trainEntropies.min(), valEntropies.min(), testEntropies.min()]).min()
maxEntropy = np.array([trainEntropies.max(), valEntropies.max(), testEntropies.max()]).max()

minEntropy = -2.62555459586
maxEntropy = -1.87461698778

# TESTING

[origImRows, origImCols, origImChannels, (trainImages, trainResults), (valImages, valResults), (testImages, testResults), fullTestImages] = loadData(2, date)
newTrainImages = resizeImages(trainImages, imRows, imCols)
newValImages = resizeImages(valImages, imRows, imCols)
newTestImages = resizeImages(testImages, imRows, imCols)

# train
trainPreds = autoencoder.predict(newTrainImages)
trainEntropies = calcCrossEntropy(newTrainImages, trainPreds, nbBins)

trainClass = trainEntropies>=minEntropy and trainEntropies<=maxEntropy

# full test

newFullTestImages = resizeImages(fullTestImages)
fullTestPreds = autoencoder.predict(newFullTestImages)

fullTestEntropies = calcCrossEntropy(newFullTestImages, fullTestPreds)

for i in range(fullTestEntropies.shape[0]):
	plt.imshow(fullTestImages[i].reshape(origImRows, origImCols), cmap='gray')
	rangeText = "Entropy range : " + minEntropy.astype(str) + " ---- " + maxEntropy.astype(str)
	text(16, 16, rangeText, color='black', fontsize=20)
	text(15, 15, rangeText, color='white', fontsize=20)
	iEntropyText = fullTestEntropies[i].astype(str)
	text(16, 31, iEntropyText, color='black', fontsize=20)
	text(15, 30, iEntropyText, color='white', fontsize=20)
	plt.show()

for i in range(fullTestPreds.shape[0]):
	plotImage = np.concatenate((newFullTestImages[i].reshape(imRows, imCols), fullTestPreds[i].reshape(imRows, imCols)), axis=1)
	plt.imshow(plotImage, cmap='gray')
	plt.show()


'''
from keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

input_img = Input(shape=(28, 28, 1))

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (32, 7, 7)

x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                nb_epoch=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))


'''
