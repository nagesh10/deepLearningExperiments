# Make sure to run the code from same directory itself,
# like :       python myKerasSmallNN_autoencoder_2layers_xxxxxx_xxxxxx.py
# instead of : python /projects/myKerasNN/myKerasSmallNN_autoencoder_2layers.py
print "myKerasSmallNN_stacked_autoencoder_2layers.py"

import numpy as np

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

import os
import sys
import time

# Paths
basePath = os.getcwd()
modelsPath = basePath + '/../MODELS'

# import stuff
sys.path.append(modelsPath)
from importStuffLoadData import *

# for plotting
import matplotlib.pyplot as plt
from pylab import text

# keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.noise import GaussianNoise
from keras.optimizers import RMSprop
from sklearn.metrics import mean_squared_error
from keras.utils import np_utils

# For using GaussianNoise (and possibly Dropout)
import tensorflow as tf
tf.python.control_flow_ops = tf

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
newFullTestImages = resizeImages(fullTestImages, imRows, imCols)

# create model
#Autoencoder
batchSize = 128
nbClasses = 1
nbEpochs = 20
nb_hidden_layers = [4096, 1024, 512, 400, ]
nb_noise = [0.3, 0.2, 0.1, ]

X_train = newTrainImages.reshape(newTrainImages.shape[0], imRows*imCols)
y_train = trainResults
X_val = newValImages.reshape(newValImages.shape[0], imRows*imCols)
y_val = valResults
X_test = newTestImages.reshape(newTestImages.shape[0], imRows*imCols)
y_test = testResults
X_fullTest = newFullTestImages.reshape(newFullTestImages.shape[0], imRows*imCols)

#Y_train = np_utils.to_categorical(y_train, nbClasses)
#Y_test = np_utils.to_categorical(y_test, nbClasses)

encoders = []
decoders = []
X_train_tmp = np.copy(X_train)
rms = RMSprop()

for i, (n_in, n_out) in enumerate(
    zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]), start=1):
    print('Training the layer {}: Input {} -> Output {}'.format(
        i, n_in, n_out))
    autoencoder = Sequential() #64x64x1
    autoencoder.add(GaussianNoise(nb_noise[i-1], input_shape=(n_in,)))
    autoencoder.add(Dense(input_dim=n_in, output_dim=n_out, activation='relu'))
    autoencoder.add(Dense(input_dim=n_out, output_dim=n_in, activation='relu'))
    autoencoder.compile(optimizer=rms, loss='mean_squared_error')
    autoencoder.fit(X_train_tmp, X_train_tmp, batch_size=batchSize, nb_epoch=nbEpochs)
    # Current encoder and decoder
    encoder = Sequential()
    encoder.add(Dense(input_dim=n_in, output_dim=n_out, weights=autoencoder.layers[1].get_weights(), activation='relu'))
    print "encoder input_dim:"
    print encoder.layers[0].input_dim
    encoders.append(encoder)
    decoder = Sequential()
    decoder.add(Dense(input_dim=n_out, output_dim=n_in, weights=autoencoder.layers[2].get_weights(), activation='relu'))
    print "decoder input_dim:"
    print decoder.layers[0].input_dim
    decoders.append(decoder)
    # Updating X_train_tmp
    X_train_tmp = encoder.predict(X_train_tmp)

print "End of training."

model = Sequential()
for encoder in encoders:
    model.add(encoder)

print "Encoders added to model."

reverseDecoders = []
for i in range(len(decoders)):
    reverseDecoders.append(decoders[-i-1])

for decoder in reverseDecoders:
    model.add(decoder)

print "Decoders added to model."
print "Model made."

'''
model.add(Dense(
    input_dim=nb_hidden_layers[-1], output_dim=nbClasses,
    activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam')
'''

timeStr = time.strftime('%Y%m%d_%H%M%S')
modelName = 'myKerasSmallNNModel_stacked_autoencoder_3layers_' + timeStr + '.h5'
model.save(modelName)
print "model saved."

'''
# Load the model
modelName = modelsPath + '/myKerasSmallNNModel_stacked_autoencoder_3layers_20161129_012638.h5'
autoencoder = load_model(modelName)
'''

# Results

trainPreds = model.predict(X_train)
trainErrors = np.zeros((trainPreds.shape[0]))
for i in range(trainPreds.shape[0]):
    trainErrors[i] = mean_squared_error(X_train[i], trainPreds[i])

print "Histogram of mean_squared_errors in training data:"
print np.histogram(trainErrors)

valPreds = model.predict(X_val)
valErrors = np.zeros((valPreds.shape[0]))
for i in range(valPreds.shape[0]):
    valErrors[i] = mean_squared_error(X_val[i], valPreds[i])

print "Histogram of mean_squared_errors in val data:"
print np.histogram(valErrors)

testPreds = model.predict(X_test)
testErrors = np.zeros((testPreds.shape[0]))
for i in range(testPreds.shape[0]):
    testErrors[i] = mean_squared_error(X_test[i], testPreds[i])

print "Histogram of mean_squared_errors in test data:"
print np.histogram(testErrors)

fullTestPreds = model.predict(X_fullTest)
fullTestErrors = np.zeros((fullTestPreds.shape[0]))
for i in range(fullTestPreds.shape[0]):
    fullTestErrors[i] = mean_squared_error(X_fullTest[i], fullTestPreds[i])

print "Histogram of mean_squared_errors in fullTest data:"
print np.histogram(fullTestErrors, bins=21)

minMSE = 0.01200463
maxMSE = 0.02600606

'''
for i in range(fullTestErrors.shape[0]):
    plotImage = np.concatenate((newFullTestImages[i].reshape(imRows, imCols), fullTestPreds[i].reshape(imRows, imCols)), axis=1)
    plt.imshow(plotImage, cmap='gray')
    rangeText = "MSE range : " + str(minMSE) + " ---- " + str(maxMSE)
    text(2, 2, rangeText, color='black', fontsize=20)
    text(2.5, 2.5, rangeText, color='white', fontsize=20)
    mseText = fullTestErrors[i].astype(str)
    text(2, 10, mseText, color='black', fontsize=20)
    text(2.5, 10.5, mseText, color='white', fontsize=20)
    plt.show()
'''
