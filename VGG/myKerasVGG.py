from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Reshape, Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy as np
import os
import cv2

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

imRows = 200
imCols = 360
emptyImagesPath = '/home/gor/projects/vikram/2016_08_19/EMPTY/EMPTIER/'
betweenImagesPath = '/home/gor/projects/vikram/2016_08_19/EMPTY/'

# load dataset
emptyImages = os.listdir(emptyImagesPath)
betweenImages = os.listdir(betweenImagesPath)

for i in betweenImages:
    #print i
    if (".bmp" not in i) or (i in emptyImages):
        #print "    removing!"  
        betweenImages.remove(i)

print len(emptyImages)
print len(betweenImages)

totalLength = len(emptyImages) + len(betweenImages)

train_inputs = np.zeros((400, 1, imRows, imCols))
train_results = np.zeros((400,3))

val_inputs = np.zeros((400, 1, imRows, imCols))
val_results = np.zeros((400,3))

test_inputs = np.zeros((400, 1, imRows, imCols))
test_results = np.zeros((400,3))

os.chdir(emptyImagesPath)
count = 0
for file in emptyImages:
    #print str(count+1) + " of " + str(totalLength)
    image = cv2.imread(emptyImagesPath+file, cv2.IMREAD_GRAYSCALE)
    #image = np.float64(np.reshape(im, (imRows*imCols)))*1./255
    image = image.reshape(1, imRows, imCols)
    #print image.shape
    if (count<200):
        val_inputs[count] = image
        val_results[count][0] = 1
    elif (count>=200 and count<400):
        test_inputs[count-200] = image
        test_results[count-200][0] = 1
    elif (count>=400 and count<600):
        train_inputs[count-400] = image
        train_results[count-400][0] = 1
    else:
        break
    count+=1

os.chdir(betweenImagesPath)
for file in betweenImages:
    #print str(count+1) + " of " + str(totalLength)
    image = cv2.imread(betweenImagesPath+file, cv2.IMREAD_GRAYSCALE)
    image = image.reshape(1, imRows, imCols)
    #print file
    #print im.shape
    #image = np.float64(np.reshape(im, (imRows*imCols)))*1./255
    #image.shape
    if (count < 800):
        train_inputs[count-600+200] = image
        train_results[count-600+200][1] = 1
    elif count>=800 and count<1000:
        val_inputs[count-800+200] = image
        val_results[count-800+200][1] = 1
    elif count>=1000 and count<1200:
        test_inputs[count-1000+200] = image
        test_results[count-1000+200][1] = 1
    else:
        break
    count+=1

val_data = (val_inputs, val_results)

print "loaded data."

'''
# encode class values as integers
encoder_train = LabelEncoder()
encoder_train.fit(train_results)
encoded_train_results = encoder_train.transform(train_results)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_train_results = np_utils.to_categorical(encoded_train_results)

# encode class values as integers
encoder_val = LabelEncoder()
encoder_val.fit(val_results)
encoded_val_results = encoder_val.transform(val_results)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_val_results = np_utils.to_categorical(encoded_val_results)

# encode class values as integers
encoder_test = LabelEncoder()
encoder_test.fit(test_results)
encoded_test_results = encoder_test.transform(test_results)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_test_results = np_utils.to_categorical(encoded_test_results)

print type(dummy_train_results)
print dummy_train_results.shape
'''

'''

# load dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

'''

'''
# create model
model = Sequential()
model.add(Dense(1024, input_dim=imRows*imCols, init='uniform', activation='relu'))
model.add(Dense(1024, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
print "model created."
'''

# create model
batchSize = 100
model = Sequential()
model.add(Convolution2D(64, 3, 3, dim_ordering='th', input_shape=(1, imRows, imCols)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(128, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(256, 3, 3))
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

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print "model compiled."

# Fit the model
model.fit(train_inputs, train_results, nb_epoch=3, batch_size=batchSize, validation_data=val_data)
print "model fitted."

# predict
#score = model.predict(test_inputs, show_accuracy=True, verbose=0)
#print "Test accuracy: " + score[1]
predictions = model.predict(test_inputs)
print "predictions made."
print type(predictions)
print predictions.shape
print predictions==test_results


# save model
model.save('/home/gor/projects/vikram/myKerasNN/my_model_5_1imRowsimCols.h5')

