# COMPLETELY WRONG CODE!!
# Max pooling pools out channelsxrows instead of rowsxcols
# Comparison formula for accuracy is wrong

print "myKerasSmallNN_1class_mse.py"
basePath = '/home/gor/projects/vikram/myKerasNN/myKerasSmallNNModel_1class_mse_20161107'
modelsPath = '/home/gor/projects/vikram/myKerasNN/MODELS/'

import numpy as np

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

import os
from sklearn import svm
import pickle
import matplotlib.pyplot as plt
from pylab import text
import sys

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
valData = (valImages, valResults)

os.chdir(basePath)

'''
[imRows, imCols, (trainImages, trainResults), trainImagesMean, valData, (testImages, testResults), fulltestImages] = loadData(nbClasses, date)


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

imRows = origImRows
imCols = origImCols
imChannels = origImChannels

# create model
#smallAlexNet
batchSize = 64
nbEpochs = 2

model = Sequential()
model.add(Convolution2D(8, 5, 5, border_mode='valid', dim_ordering='th', input_shape=(1, imRows, imCols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dense(nbClasses))
model.add(Activation('sigmoid'))

# Compile the model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
print "model compiled."

# Fit the model
model.fit(trainImages, trainResults, nb_epoch=nbEpochs, batch_size=batchSize, validation_data=valData, verbose=1)
print "model fitted."

# Save the model
modelName = 'myKerasSmallNNModel_1class_mse_' + timeStr + '.h5'
model.save(modelName)
print "model saved."

'''
# Load the model
modelName = modelsPath + 'myKerasSmallNNModel_binaryCE_20161103_192854.h5'
model = load_model(modelName)
'''

[imRows, imCols, (trainImages, trainResults), (valImages, valResults), (testImages, testResults), fullTestImages] = loadData(2, date)

# Training Accuracy
trainPreds = model.predict(trainImages)
nOfCorrectTrainPreds = (np.round(trainPreds)==trainResults[:,0]).sum()/2
nOfTrainImages = trainPreds.shape[0]
trainAccuracy = nOfCorrectTrainPreds.astype(float)/nOfTrainImages
print "Training accuracy = "
print trainAccuracy

# Val data accuracy
valPreds = model.predict(valImages)
nOfCorrectValPreds = (np.round(valPreds)==valResults).sum()/2
nOfValImages = valPreds.shape[0]
valAccuracy = nOfCorrectValPreds.astype(float)/nOfValImages
print "Validation accuracy = "
print valAccuracy

# Test data accuracy
testPreds = model.predict(testImages)
nOfCorrectTestPreds = (np.round(testPreds)==testResults).sum()/2
nOfTestImages = testPreds.shape[0]
testAccuracy = nOfCorrectTestPreds.astype(float)/nOfTestImages
print "Test data accuracy = "
print testAccuracy

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
	'''
	if trainResults1[i]==1:
		rText = 'r->  BETWEEN'
	else:
		rText = 'r-> EMPTY'
	'''
	plt.imshow(image.reshape(imRows, imCols), cmap='gray')
	text(16, 16, imageText, color='black', fontsize=20)
	text(15, 15, imageText, color='white', fontsize=20)
	text(16, 31, imageScore.astype(str), color='black', fontsize=20)
	text(15, 30, imageScore.astype(str), color='white', fontsize=20)
	#text(16, 61, rText, color='black', fontsize=20)
	#text(15, 60, rText, color='white', fontsize=20)
	plt.show()
