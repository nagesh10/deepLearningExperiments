print "myKerasNN.py"

import numpy as np
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

from importStuffLoadData import *

nbClasses = 2
#date = '2016_08_19'
date = '2016_11_03'

[imRows, imCols, (trainImages, trainResults), trainImagesMean, valData, (testImages, testResults), fulltestImages] = loadData(nbClasses, date)

'''
# create model
model = Sequential()
model.add(Dense(1024, input_dim=imRows*imCols, init='uniform', activation='relu'))
model.add(Dense(1024, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
print "model created."
'''

# create model
#AlexNet
batchSize = 64
nbEpochs = 50

model = Sequential()
model.add(Convolution2D(96, 11, 11, border_mode='valid', dim_ordering='th', input_shape=(1, imRows, imCols), subsample=(4,4)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(256, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(384, 3, 3))
model.add(Activation('relu'))

model.add(Convolution2D(384, 3, 3))
model.add(Activation('relu'))

model.add(Convolution2D(256, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(512))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))

'''
# Compile model
opt = SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print "model compiled."
'''

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(trainImages, trainResults, nb_epoch=nbEpochs, batch_size=batchSize, validation_data=valData, verbose=1)
print "model fitted."

# Save the model
timeStr = time.strftime('%Y%m%d_%H%M%S')
modelName = '/home/gor/projects/vikram/myKerasNN/myKerasNNModel_' + timeStr + '.h5'
model.save(modelName)
