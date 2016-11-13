print "myKerasSmallNN_WithSVM_2classes.py"
basePath = '/home/gor/projects/vikram/myKerasNN/myKerasSmallNNModel_withSVM_2classes_20161107'
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
from importStuffLoadData import *

# Save time
timeStr = time.strftime('%Y%m%d_%H%M%S')

nbClasses = 2
#date = '2016_08_19'
date = '2016_11_03'

[imRows, imCols, (trainImages, trainResults), (valImages, valResults), (testImages, testResults), fullTestImages] = loadData(nbClasses, date)
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

'''
# create model
model = Sequential
#smallAlexNet
batchSize = 64
nbEpochs = 100

model = Sequential()
model.add(Convolution2D(32, 11, 11, border_mode='valid', dim_ordering='th', input_shape=(1, imRows, imCols), subsample=(4,4)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 5, 5, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(nbClasses))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print "model compiled."

# Fit the model
model.fit(trainImages, trainResults, nb_epoch=nbEpochs, batch_size=batchSize, validation_data=valData, verbose=1)
print "model fitted."

# Save the model
modelName = '/home/gor/projects/vikram/myKerasNN/myKerasSmallNNModel_SVM_' + timeStr + '.h5'
model.save(modelName)
print "model saved."
'''

## MAKE SUB-MODEL + scikit.learn.SVM

modelName = modelsPath + 'myKerasSmallNNModel_binaryCE_20161103_192854.h5'
model = load_model(modelName)
print "model loaded."

# subModel
subModel = Sequential()
subModel.add(Convolution2D(32, 11, 11, border_mode='valid', dim_ordering='th', input_shape=(1, imRows, imCols), subsample=(4,4), weights=model.layers[0].get_weights()))
subModel.add(Activation('relu'))
subModel.add(MaxPooling2D(pool_size=(2, 2)))

subModel.add(Convolution2D(32, 5, 5, weights=model.layers[3].get_weights()))
subModel.add(Activation('relu'))
subModel.add(MaxPooling2D(pool_size=(2, 2)))

subModel.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
subModel.add(Dense(64, weights=model.layers[7].get_weights()))
subModel.add(Activation('relu'))
#model.add(Dropout(0.5))
subModel.add(Dense(64, weights=model.layers[9].get_weights()))
subModel.add(Activation('relu'))

print "subModel created"

# Calculating activations for data
subModelTrainActivations = subModel.predict(trainImages)
subModelValActivations = subModel.predict(valImages)
subModelTestActivations = subModel.predict(testImages)

print "activations from subModel calculated"

# Single row results
trainResults1 = trainResults[:, 1]
valResults1 = valResults[:, 1]
testResults1 = testResults[:, 1]

'''
# SVM
mySvm = svm.LinearSVC()

# Fit the SVM with 2 classes
mySvm.fit(subModelTrainActivations, trainResults1)

print "SVM fitted"

# Save SVM model
svmFileName = 'mySVMmodel_2classes_' + timeStr + '.pkl'
pickle.dump(mySvm, open(svmFileName, 'wb'))

print "SVM model saved as " + svmFileName
'''

# Load SVM model
svmFileName = modelsPath + 'mySVMmodel_2classes_20161107_221449.pkl'
mySvm = pickle.load(open(svmFileName, 'rb'))

# Training Accuracy
svmTrainPreds = mySvm.predict(subModelTrainActivations)
svmTrainScores = mySvm.decision_function(subModelTrainActivations)
nOfCorrectTrainPreds = (svmTrainPreds==trainResults1).sum()
nOfTrainImages = subModelTrainActivations.shape[0]
trainAccuracy = nOfCorrectTrainPreds.astype(float)/nOfTrainImages
print "Training accuracy = "
print trainAccuracy

'''
np.histogram(svmTrainScores, bins=32, range=(-16, 16))
(array([  0,    2,    8,    9,   15,   10,   22,   28,   23,   12,   10,
         16,   26,   15,   21,    2,    2,   24,   34,   50,   63,   96,
        187,  163,  127,   72,   69,   62,   48,   20,    4,    2]),
array([-16., -15., -14., -13., -12., -11., -10.,  -9.,  -8.,  -7.,  -6.,
        -5.,  -4.,  -3.,  -2.,  -1.,   0.,   1.,   2.,   3.,   4.,   5.,
         6.,   7.,   8.,   9.,  10.,  11.,  12.,  13.,  14.,  15.,  16.]))
'''

# Val data accuracy
svmValPreds = mySvm.predict(subModelValActivations)
svmValScores = mySvm.decision_function(subModelValActivations)
nOfCorrectValPreds = (svmValPreds==valResults1).sum()
nOfValImages = subModelValActivations.shape[0]
valAccuracy = nOfCorrectValPreds.astype(float)/nOfValImages
print "Validation accuracy = "
print valAccuracy

# Test data accuracy
svmTestPreds = mySvm.predict(subModelTestActivations)
svmTestScores = mySvm.decision_function(subModelTestActivations)
nOfCorrectTestPreds = (svmTestPreds==testResults1).sum()
nOfTestImages = subModelTestActivations.shape[0]
testAccuracy = nOfCorrectTestPreds.astype(float)/nOfTestImages
print "Test data accuracy = "
print testAccuracy

# Testing trainImages
for i in range(trainImages.shape[0]):
	image = trainImages[i].reshape(1, 1, imRows, imCols)
	imageClass = mySvm.predict(subModel.predict(image))
	imageScore = mySvm.decision_function(subModel.predict(image))
	if imageClass==1:
		imageText = 'svm->BETWEEN'
	else:
		imageText = 'svm->EMPTY'
	if trainResults1[i]==1:
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
	imageClass = mySvm.predict(subModel.predict(image))
	imageScore = mySvm.decision_function(subModel.predict(image))
	if imageClass==1:
		imageText = 'svm->BETWEEN'
	else:
		imageText = 'svm->EMPTY'
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

##### MAJOR FAILURE for caskets with packet
