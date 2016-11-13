print "myKerasNN.py"

import numpy as np
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

from importStuffLoadData import *

nbClasses = 2

[imRows, imCols, (trainImages, trainResults), valData, (testImages, testResults), fulltestImages] = loadData(nbClasses)

# Find trainnImagesMean
trainImagesMean = trainImages.mean(axis=0)

# Save trainImagesMean
timeStr = time.strftime('%Y%m%d_%H%M%S')
trainImagesMeanFileName = 'trainImagesMean_'+timeStr+'.npy'
np.save(trainImagesMeanFileName, trainImagesMean)
print "saved " + trainImagesMeanFileName

# Subtract trainImagesMean from trainingImages and valImages
trainImages-=trainImagesMean
(valImages, valResults) = valData
valImages-=trainImagesMean
valData = (valImages, valResults)

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
batchSize = 64
nbEpochs = 50

model = Sequential()
model.add(Convolution2D(32, 11, 11, border_mode='valid', dim_ordering='th', input_shape=(1, imRows, imCols), subsample=(4,4)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 5, 5))
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
'''
# Compile model
opt = SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print "model compiled."
'''

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print "model compiled."

# Fit the model
model.fit(trainImages, trainResults, nb_epoch=nbEpochs, batch_size=batchSize, validation_data=valData, verbose=1)
print "model fitted."

# Save the model
modelName = '/home/gor/projects/vikram/myKerasNN/myKerasSmallNNModel_' + timeStr + '.h5'
model.save(modelName)
print "model saved."
