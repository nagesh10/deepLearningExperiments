print "myKerasSmallNN_autoencoder.py"
basePath = '/home/gor/projects/vikram/myKerasNN/myKerasSmallNNModel_autoencoder_2layers'
modelsPath = '/home/gor/projects/vikram/myKerasNN/MODELS/'

import numpy as np

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pylab import text
from scipy.misc import imresize as imresize
import sys
import time

from keras.models import Sequential, load_model
from keras.layers import Dense, Convolution2D, MaxPooling2D, UpSampling2D
from sklearn.metrics import mean_squared_error

sys.path.append(modelsPath)
from importStuffLoadData import loadData

# Save time
timeStr = time.strftime('%Y%m%d_%H%M%S')

nbClasses = 1
#date = '2016_08_19'
date = '2016_11_03'

[origImRows, origImCols, (trainImages, trainResults), (valImages, valResults), (testImages, testResults), fullTestImages] = loadData(nbClasses, date)
valData = (valImages, valResults)

os.chdir(basePath)

# TODO: KL transform
# TODO: Covariance Equalization

# imresize
imRows = 64
imCols = 64

newTrainImages = np.zeros((trainImages.shape[0], imRows, imCols, 1))
for i in range(trainImages.shape[0]):
	newTrainImages[i] = imresize(trainImages[i].reshape(origImRows, origImCols), (imRows, imCols)).reshape(imRows, imCols, 1)/255.0


newValImages = np.zeros((valImages.shape[0], imRows, imCols, 1))
for i in range(valImages.shape[0]):
	newValImages[i] = imresize(valImages[i].reshape(origImRows, origImCols), (imRows, imCols)).reshape(imRows, imCols, 1)/255.0


newTestImages = np.zeros((testImages.shape[0], imRows, imCols, 1))
for i in range(testImages.shape[0]):
	newTestImages[i] = imresize(testImages[i].reshape(origImRows, origImCols), (imRows, imCols)).reshape(imRows, imCols, 1)/255.0


# create model
#Autoencoder
batchSize = 128
nbEpochs = 100

autoencoder = Sequential() #64x64x1
autoencoder.add(Convolution2D(32, 11, 11, input_shape=(imRows, imCols, 1), border_mode='same',  activation='relu')) #64x64x16
autoencoder.add(MaxPooling2D(pool_size=(2, 2))) #32x32x16
autoencoder.add(Convolution2D(16, 7, 7, border_mode='same', activation='relu')) #32x32x8
autoencoder.add(MaxPooling2D(pool_size=(2, 2))) #32x32x16
autoencoder.add(Convolution2D(8, 3, 3, border_mode='same', activation='relu')) #32x32x8
autoencoder.add(MaxPooling2D(pool_size=(2, 2))) #16x16x8

# at this point the representation is (16, 16, 8)

autoencoder.add(Convolution2D(8, 3, 3, border_mode='same', activation='relu')) #16x16x8
autoencoder.add(UpSampling2D(size=(2, 2))) #32x32x8
autoencoder.add(Convolution2D(16, 7, 7, border_mode='same', activation='relu')) #32x32x16
autoencoder.add(UpSampling2D(size=(2, 2))) #32x32x8
autoencoder.add(Convolution2D(32, 11, 11, border_mode='same', activation='relu')) #32x32x16
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
# Load the model
modelName = modelsPath + 'myKerasSmallNNModel_autoencoder_2layers_20161108_192854.h5'
autoencoder = load_model(modelName)
'''

# FINDING RANGE OF ENTROPIES

rowTrainImages = newTrainImages.reshape(newTrainImages.shape[0], imRows*imCols)
rowValImages = newValImages.reshape(newValImages.shape[0], imRows*imCols)
rowTestImages = newTestImages.reshape(newTestImages.shape[0], imRows*imCols)

# predict for train

trainPreds = autoencoder.predict(newTrainImages)
rowTrainPreds = trainPreds.reshape(trainPreds.shape[0], imRows*imCols)

'''
for i in range(trainPreds.shape[0]):
	plotImage = np.concatenate((newTrainImages[i].reshape(imRows, imCols), trainPreds[i].reshape(imRows, imCols)), axis=1)
	plt.imshow(plotImage, cmap='gray')
	plt.show()
'''

# calc entropy for train
nbBins = 12

probRowTrainImages = np.zeros((rowTrainImages.shape[0], nbBins))
for i in range(rowTrainImages.shape[0]):
	probRowTrainImages[i] = np.bincount((rowTrainImages[i]*10).astype(int), minlength=nbBins)/float(imRows*imCols)

probRowTrainPreds = np.zeros((rowTrainPreds.shape[0], nbBins))
for i in range(rowTrainPreds.shape[0]):
	probRowTrainPreds[i] = np.bincount((rowTrainPreds[i]*10).astype(int), minlength=nbBins)/float(imRows*imCols)

trainEntropies = np.zeros((rowTrainPreds.shape[0], 1))
for i in range(rowTrainPreds.shape[0]):
	nonZeroCols = np.nonzero(probRowTrainPreds[i])
	trainEntropies[i] = (probRowTrainImages[i][nonZeroCols]*np.log(probRowTrainPreds[i][nonZeroCols]) +
        (1-probRowTrainImages[i][nonZeroCols])*np.log(1-probRowTrainPreds[i][nonZeroCols])).sum()

# predict, calc entropy for val

valPreds = autoencoder.predict(newValImages)
rowValPreds = valPreds.reshape(valPreds.shape[0], imRows*imCols)

probRowValImages = np.zeros((rowValImages.shape[0], nbBins))
for i in range(rowValImages.shape[0]):
	probRowValImages[i] = np.bincount((rowValImages[i]*10).astype(int), minlength=nbBins)/float(imRows*imCols)

probRowValPreds = np.zeros((rowValPreds.shape[0], nbBins))
for i in range(rowValPreds.shape[0]):
	probRowValPreds[i] = np.bincount((rowValPreds[i]*10).astype(int), minlength=nbBins)/float(imRows*imCols)

valEntropies = np.zeros((rowValPreds.shape[0], 1))
for i in range(rowValPreds.shape[0]):
	nonZeroCols = np.nonzero(probRowValPreds[i])
	valEntropies[i] = (probRowValImages[i][nonZeroCols]*np.log(probRowValPreds[i][nonZeroCols]) +
        (1-probRowValImages[i][nonZeroCols])*np.log(1-probRowValPreds[i][nonZeroCols])).sum()

# predict, calc entropy for test

testPreds = autoencoder.predict(newTestImages)
rowTestPreds = testPreds.reshape(testPreds.shape[0], imRows*imCols)

probRowTestImages = np.zeros((rowTestImages.shape[0], nbBins))
for i in range(rowTestImages.shape[0]):
	probRowTestImages[i] = np.bincount((rowTestImages[i]*10).astype(int), minlength=nbBins)/float(imRows*imCols)

probRowTestPreds = np.zeros((rowTestPreds.shape[0], nbBins))
for i in range(rowTestPreds.shape[0]):
	probRowTestPreds[i] = np.bincount((rowTestPreds[i]*10).astype(int), minlength=nbBins)/float(imRows*imCols)

testEntropies = np.zeros((rowTestPreds.shape[0], 1))
for i in range(rowTestPreds.shape[0]):
	nonZeroCols = np.nonzero(probRowTestPreds[i])
	testEntropies[i] = (probRowTestImages[i][nonZeroCols]*np.log(probRowTestPreds[i][nonZeroCols]) +
        (1-probRowTestImages[i][nonZeroCols])*np.log(1-probRowTestPreds[i][nonZeroCols])).sum()

# Entropies

print "trainEntropies : " + trainEntropies.min().astype(str) + " ---- " + trainEntropies.max().astype(str)
print "valEntropies : " + valEntropies.min().astype(str) + " ---- " + valEntropies.max().astype(str)
print "testEntropies : " + testEntropies.min().astype(str) + " ---- " + testEntropies.max().astype(str)

minEntropy = np.array([trainEntropies.min(), valEntropies.min(), testEntropies.min()]).min()
maxEntropy = np.array([trainEntropies.max(), valEntropies.max(), testEntropies.max()]).max()

'''
# Full test

newFullTestImages = np.zeros((fullTestImages.shape[0], imRows, imCols, 1))
rowFullTestImages = np.zeros((fullTestImages.shape[0], imRows*imCols))
for i in range(fullTestImages.shape[0]):
	newFullTestImages[i] = imresize(fullTestImages[i].reshape(fullTestImages.shape[2], fullTestImages.shape[3]), (imRows, imCols)).reshape(imRows, imCols, 1)/255.0
	rowFullTestImages[i] = newFullTestImages[i].reshape(imRows*imCols)/255.0

fullTestPreds = autoencoder.predict(newFullTestImages)
rowFullTestPreds = fullTestPreds.reshape(fullTestPreds.shape[0], imRows*imCols)

fullTestEntropies = np.zeros(rowFullTestPreds.shape[0],)
for i in range(rowFullTestPreds.shape[0]):
	nonZeroCols = np.nonzero(probRowFullTestPreds[i])
	fullTestEntropies[i] = probRowFullTestImages[i][nonZeroCols]*np.log(probRowFullTestPreds[i][nonZeroCols]) + (1-probRowFullTestImages[i][nonZeroCols])*np.log(1-probRowFullTestPreds[i][nonZeroCols])

for i in range(fullTestEntropies.shape[0]):
	plt.imshow(fullTestImages[i].reshape(origImRows, origImCols), cmap='gray')
	rangeText = "Entropy range : " + minEntropy.astype(str) + " - " + maxEntropy.astype(str)
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
