# DOESN'T WORK - HORRIBLE ACCURACY
# Training is awesome, but even between_packets are matching with train.
# Pulling everything into the trained class.

print "myKerasSmallNN_1class_binaryCE.py"
basePath = '/home/gor/projects/vikram/myKerasNN/myKerasSmallNNModel_1class/'
modelsPath = '/home/gor/projects/vikram/myKerasNN/MODELS/'

import numpy as np

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

import os
import matplotlib.pyplot as plt
from pylab import text
from scipy.misc import imresize as imresize
import sys
import time

sys.path.append(modelsPath)
from importStuffLoadData import loadData

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten

# Save time
timeStr = time.strftime('%Y%m%d_%H%M%S')

nbClasses = 1
#date = '2016_08_19'
date = '2016_11_03'

[origImRows, origImCols, origImChannels, (trainImages, trainResults), (valImages, valResults), (testImages, testResults), fullTestImages] = loadData(nbClasses, date)

os.chdir(basePath)

'''
[origImRows, origImCols, (trainImages, trainResults), trainImagesMean, valData, (testImages, testResults), fulltestImages] = loadData(nbClasses, date)


# Find trainnImagesMean
trainImagesMean = trainImages.mean(axis=0)

# Save trainImagesMean
trainImagesMeanFileName = basePath+'trainImagesMean_'+timeStr+'.npy'
np.save(trainImagesMeanFileName, trainImagesMean)
print "saved " + trainImagesMeanFileName

# Subtract trainImagesMean from trainingImages and valImages
trainImages-=trainImagesMean
(valImages, valResults) = valData
valImages-=trainImagesMean
valData = (valImages, valResults)
'''

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
#smallAlexNet
batchSize = 128
nbEpochs = 5

model = Sequential() #64x64x1
model.add(Convolution2D(2, 7, 7, input_shape=(imRows, imCols, 1), activation='relu')) #54x54x2
model.add(MaxPooling2D(pool_size=(2, 2))) #27x27x2

model.add(Flatten()) #1458
model.add(Dense(8, activation='relu'))
model.add(Dense(nbClasses, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print "model compiled."

# Fit the model
model.fit(newTrainImages, trainResults, nb_epoch=nbEpochs, batch_size=batchSize, validation_data=(newValImages, valResults), verbose=1)
print "model fitted."

# Save the model
timeStr = time.strftime('%Y%m%d_%H%M%S')
modelName = 'myKerasSmallNNModel_1class_binaryCE_' + timeStr + '.h5'
model.save(modelName)
print "model saved."

'''
# Load the model
modelName = modelsPath + 'myKerasSmallNNModel_1class_binaryCE_20161112_175216.h5'
model = load_model(modelName)
'''

trainPreds = model.predict(newTrainImages)
valPreds = model.predict(newValImages)
testPreds = model.predict(newTestImages)
preds = np.concatenate([np.concatenate([trainPreds, valPreds]), testPreds])
predsMin = preds.min()

# Load all data

[origImRows, origImCols, (trainImages, trainResults), (valImages, valResults), (testImages, testResults), fullTestImages] = loadData(2, date)

newTrainImages = np.zeros((trainImages.shape[0], imRows, imCols, 1))
for i in range(trainImages.shape[0]):
	newTrainImages[i] = imresize(trainImages[i].reshape(origImRows, origImCols), (imRows, imCols)).reshape(imRows, imCols, 1)/255.0


newValImages = np.zeros((valImages.shape[0], imRows, imCols, 1))
for i in range(valImages.shape[0]):
	newValImages[i] = imresize(valImages[i].reshape(origImRows, origImCols), (imRows, imCols)).reshape(imRows, imCols, 1)/255.0


newTestImages = np.zeros((testImages.shape[0], imRows, imCols, 1))
for i in range(testImages.shape[0]):
	newTestImages[i] = imresize(testImages[i].reshape(origImRows, origImCols), (imRows, imCols)).reshape(imRows, imCols, 1)/255.0


# Training Accuracy
trainPreds = model.predict(newTrainImages)
nOfCorrectTrainPreds = ((trainPreds>=predsMin).astype(int)==trainResults[:,0].reshape(trainResults.shape[0], 1)).sum()
nOfTrainImages = trainPreds.shape[0]
trainAccuracy = nOfCorrectTrainPreds.astype(float)/nOfTrainImages
print "Training accuracy = "
print trainAccuracy

# Val data accuracy
valPreds = model.predict(newValImages)
nOfCorrectValPreds = ((valPreds>=predsMin).astype(int)==(valResults[:,0].reshape(valResults.shape[0], 1))).sum()
nOfValImages = valPreds.shape[0]
valAccuracy = nOfCorrectValPreds.astype(float)/nOfValImages
print "Validation accuracy = "
print valAccuracy

# Test data accuracy
testPreds = model.predict(newTestImages)
nOfCorrectTestPreds = ((testPreds>=predsMin).astype(int)==(testResults[:,0].reshape(testResults.shape[0], 1))).sum()
nOfTestImages = testPreds.shape[0]
testAccuracy = nOfCorrectTestPreds.astype(float)/nOfTestImages
print "Test data accuracy = "
print testAccuracy

'''
# Testing trainImages
for i in range(trainImages.shape[0]):
	image = trainImages[i].reshape(1, 1, imRows, imCols)
	imageScore = model.predict(image)
	if np.round(imageScore)==0:
		imageText = 'NN->EMPTY'
	else:
		imageText = 'NN->BETWEEN'
	if trainResults[i]==1:
		rText = 'r->  BETWEEN'
	else:
		rText = 'r-> EMPTY'
	plt.imshow(image.reshape(imRows, imCols), cmap='gray')
	text(16, 16, imageText, color='black', fontsize=20)
	text(15, 15, imageText, color='white', fontsize=20)
	text(16, 31, imageScore.astype(str), color='black', fontsize=20)
	text(15, 30, imageScore.astype(str), color='white', fontsize=20)
	text(16, 61, rText, color='black', fontsize=20)
	text(15, 60, rText, color='white', fontsize=20)
	plt.show()


# Testing fullTestImages
for i in range(fullTestImages.shape[0]):
	image = fullTestImages[i].reshape(1, 1, imRows, imCols)
	imageScore = model.predict(image)
	if np.round(imageScore)==1:
		imageText = 'NN->BETWEEN'
	else:
		imageText = 'NN->EMPTY'
	plt.imshow(image.reshape(imRows, imCols), cmap='gray')
	text(16, 16, imageText, color='black', fontsize=20)
	text(15, 15, imageText, color='white', fontsize=20)
	text(16, 31, imageScore.astype(str), color='black', fontsize=20)
	text(15, 30, imageScore.astype(str), color='white', fontsize=20)
	plt.show()
'''
