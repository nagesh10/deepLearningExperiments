print "myKerasNN.py"
basePath = '/home/gor/projects/vikram/myKerasNN/myKerasSmallNNModel_2classes_binaryCE/'
modelsPath = '/home/gor/projects/vikram/myKerasNN/MODELS/'

import numpy as np
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

import os
import sys
import time

sys.path.append(modelsPath)
from importStuffLoadData import loadData

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten

# Save time
timeStr = time.strftime('%Y%m%d_%H%M%S')

nbClasses = 2
#date = '2016_08_19'
date = '2016_11_03'

[origImRows, origImCols, origImChannels, (trainImages, trainResults), (valImages, valResults), (testImages, testResults), fulltestImages] = loadData(nbClasses, date)
valData = (valImages, valResults)

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

'''
# create model
model = Sequential()
model.add(Dense(1024, input_dim=imRows*imCols, init='uniform', activation='relu'))
model.add(Dense(1024, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
print "model created."
'''

# create model
#smallAlexNet
batchSize = 128
nbEpochs = 100

model = Sequential() #200x360x1
model.add(Convolution2D(32, 11, 11, input_shape=(origImRows, origImCols, origImChannels), activation='relu')) #190x350x32
model.add(MaxPooling2D(pool_size=(2, 2))) #95x175x32

model.add(Convolution2D(16, 7, 7, activation='relu')) #89x169x16
model.add(MaxPooling2D(pool_size=(2, 2))) #44x84x16

model.add(Convolution2D(8, 3, 3, activation='relu')) #42x82x8
model.add(MaxPooling2D(pool_size=(2, 2))) #21x41x8

model.add(Flatten())  #6888
model.add(Dense(512, activation='relu')) #512
#model.add(Dropout(0.5))
model.add(Dense(64, activation='relu')) #64
model.add(Dense(nbClasses)) #nbClasses
model.add(Activation('softmax'))

'''
# Compile model
opt = SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print "model compiled."
'''

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print "model compiled."

# Fit the model
model.fit(trainImages, trainResults, nb_epoch=nbEpochs, batch_size=batchSize, shuffle=True, validation_data=valData, verbose=1)
print "model fitted."

# Save the model
timeStr = time.strftime('%Y%m%d_%H%M%S')
modelName = 'myKerasSmallNNModel_2classes_binaryCE_' + timeStr + '.h5'
model.save(modelName)
print "model saved."

'''
# Load the model
modelName = modelsPath + 'myKerasSmallNNModel_2classes_binaryCE_20161111_211603.h5'
model = load_model(modelName)
'''

# Training Accuracy
trainPreds = model.predict(trainImages)
nOfCorrectTrainPreds = (np.round(trainPreds)==trainResults).sum()/2
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
