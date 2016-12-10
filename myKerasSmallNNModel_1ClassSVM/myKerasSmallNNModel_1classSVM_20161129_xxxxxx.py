# Make sure to run the code from same directory itself,
# like :       python myKerasSmallNN_autoencoder_2layers_xxxxxx_xxxxxx.py
# instead of : python /projects/myKerasNN/myKerasSmallNN_autoencoder_2layers.py
print "myKerasSmallNN_1classSVM.py"

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
from sklearn import svm

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


# imresize
imRows = 100
imCols = 100
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
nb_hidden_layers = [imRows*imCols, 4096, 1024, 512, 200, ]
nb_noise = [0.35, 0.3, 0.2, 0.1, ]

X_train = newTrainImages.reshape(newTrainImages.shape[0], imRows*imCols)
y_train = trainResults
X_val = newValImages.reshape(newValImages.shape[0], imRows*imCols)
y_val = valResults
X_test = newTestImages.reshape(newTestImages.shape[0], imRows*imCols)
y_test = testResults
X_fullTest = newFullTestImages.reshape(newFullTestImages.shape[0], imRows*imCols)

# create model
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
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train_tmp, X_train_tmp, batch_size=batchSize, nb_epoch=nbEpochs)
    # Current encoder and decoder
    encoder = Sequential()
    encoder.add(Dense(input_dim=n_in, output_dim=n_out, weights=autoencoder.layers[1].get_weights(), activation='relu'))
    encoders.append(encoder)
    decoder = Sequential()
    decoder.add(Dense(input_dim=n_out, output_dim=n_in, weights=autoencoder.layers[2].get_weights(), activation='relu'))
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

model.compile(loss='mse', optimizer='adam')

trainPreds = model.predict(X_train)

# SVM
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)

timeStr = time.strftime('%Y%m%d_%H%M%S')
modelName = 'myKerasSmallNNModel_stacked_autoencoder_3layers_' + timeStr + '.h5'
model.save(modelName)
print "model saved."

'''
# Load the model
modelName = modelsPath + '/myKerasSmallNNModel_stacked_autoencoder_3layers_20161129_012638.h5'
autoencoder = load_model(modelName)
'''





# Save the model
modelName = 'myKerasSmallNNModel_1classSVM_' + timeStr + '.h5'
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
