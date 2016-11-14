print "importStuffLoadData.py"

import cv2
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder
from scipy.misc import imresize as imresize

#return [imRows, imCols, trainData, valData, testData, fullTestImages]
def loadData (nbClasses, date, norm0to1=True):

    imRows = 200
    imCols = 360
    imChannels = 1

    if (date=='2016_08_19'):
        emptyImagesPath = '/home/gor/projects/vikram/2016_08_19/Train/EMPTY/'
        betweenImagesPath = '/home/gor/projects/vikram/2016_08_19/Train/BETWEEN/'
        print 'emptyImagesPath=' + emptyImagesPath
        print 'betweenImagesPath=' + betweenImagesPath

    elif (date=='2016_11_03'):
        emptyImagesPath = '/home/gor/projects/vikram/2016_11_03/EMPTY/'
        betweenImagesPath = '/home/gor/projects/vikram/2016_11_03/BETWEEN/'
        print 'emptyImagesPath=' + emptyImagesPath
        print 'betweenImagesPath=' + betweenImagesPath

    fullImagesPath = '/home/gor/projects/vikram/2016_08_19/FULL/'

    # load dataset
    emptyImages = sorted(os.listdir(emptyImagesPath))
    for i in emptyImages:
        if ".bmp" not in i:
            emptyImages.remove(i)

    betweenImages = sorted(os.listdir(betweenImagesPath))
    for i in betweenImages:
        if ".bmp" not in i:
            betweenImages.remove(i)

    print len(emptyImages)
    print len(betweenImages)

    # Use 1/2 for training, 1/4 for validation, 1/4 for test
    trainEmptyImages = emptyImages[:len(emptyImages)/2]
    trainBetweenImages = betweenImages[:len(betweenImages)/2]
    nOfTrainImages = len(trainEmptyImages)
    if nbClasses>1:
        nOfTrainImages += len(trainBetweenImages)
    print "nOfTrainImages = " + str(nOfTrainImages)

    valEmptyImages = emptyImages[len(emptyImages)/2:len(emptyImages)*3/4]
    valBetweenImages = betweenImages[len(betweenImages)/2:len(betweenImages)*3/4]
    nOfValImages = len(valEmptyImages)
    if nbClasses>1:
        nOfValImages += len(valBetweenImages)
    print "nOfValImages = " + str(nOfValImages)

    testEmptyImages = emptyImages[len(emptyImages)*3/4:]
    testBetweenImages = betweenImages[len(betweenImages)*3/4:]
    nOfTestImages = len(testEmptyImages)
    if nbClasses>1:
        nOfTestImages += len(testBetweenImages)
    print "nOfTestImages = " + str(nOfTestImages)

    trainImages = np.zeros((nOfTrainImages, imRows, imCols, imChannels))
    trainResults = np.zeros((nOfTrainImages, nbClasses))

    valImages = np.zeros((nOfValImages, imRows, imCols, imChannels))
    valResults = np.zeros((nOfValImages, nbClasses))

    testImages = np.zeros((nOfTestImages, imRows, imCols, imChannels))
    testResults = np.zeros((nOfTestImages, nbClasses))


    # Read training images
    count = 0
    os.chdir(emptyImagesPath)
    for file in trainEmptyImages:
        #print str(count+1) + " of " + str(len(emptyImages)) + "empty"
        image = cv2.imread(emptyImagesPath+file, cv2.IMREAD_GRAYSCALE) ###GRAYSCALE!!!
        #image = np.float64(np.reshape(im, (imRows*imCols)))*1./255
        image = image.reshape(imRows, imCols, imChannels).astype("float32")
        if norm0to1:
            image/=255
        trainImages[count] = image
        trainResults[count][0] = 1   # idx 0 for empty
        count+=1

    if nbClasses>1:
        os.chdir(betweenImagesPath)
        for file in trainBetweenImages:
            #print str(count+1) + " of " + str(len(betweenImages)) + "between"
            image = cv2.imread(betweenImagesPath+file, cv2.IMREAD_GRAYSCALE) ###GRAYSCALE!!!
            #image = np.float64(np.reshape(im, (imRows*imCols)))*1./255
            image = image.reshape(imRows, imCols, imChannels).astype("float32")
            if norm0to1:
                image/=255
            trainImages[count] = image
            trainResults[count][1] = 1   # idx 1 for between
            count+=1

    randomOrder = np.random.choice(nOfTrainImages, size=nOfTrainImages, replace=False)
    trainImages = trainImages[randomOrder]
    trainResults = trainResults[randomOrder]

    trainData = (trainImages, trainResults)

    print "trainData loaded"

    # Read validation images
    count = 0
    os.chdir(emptyImagesPath)
    for file in valEmptyImages:
        #print str(count+1) + " of " + str(totalLength)
        image = cv2.imread(emptyImagesPath+file, cv2.IMREAD_GRAYSCALE) ###GRAYSCALE!!!
        #image = np.float64(np.reshape(im, (imRows*imCols)))*1./255
        image = image.reshape(imRows, imCols, imChannels).astype("float32")
        if norm0to1:
            image/=255
        valImages[count] = image
        valResults[count][0] = 1   # idx 0 for empty
        count+=1

    if nbClasses>1:
        os.chdir(betweenImagesPath)
        for file in valBetweenImages:
            #print str(count+1) + " of " + str(totalLength)
            image = cv2.imread(betweenImagesPath+file, cv2.IMREAD_GRAYSCALE) ###GRAYSCALE!!!
            #image = np.float64(np.reshape(im, (imRows*imCols)))*1./255
            image = image.reshape(imRows, imCols, imChannels).astype("float32")
            if norm0to1:
                image/=255
            valImages[count] = image
            valResults[count][1] = 1   # idx 1 for between
            count+=1

    randomOrder = np.random.choice(nOfValImages, size=nOfValImages, replace=False)
    valImages = valImages[randomOrder]
    valResults = valResults[randomOrder]

    valData = (valImages, valResults)

    print "valData loaded"

    # Read test images
    count = 0
    os.chdir(emptyImagesPath)
    for file in testEmptyImages:
        #print str(count+1) + " of " + str(totalLength)
        image = cv2.imread(emptyImagesPath+file, cv2.IMREAD_GRAYSCALE) ###GRAYSCALE!!!
        #image = np.float64(np.reshape(im, (imRows*imCols)))*1./255
        image = image.reshape(imRows, imCols, imChannels).astype("float32")
        if norm0to1:
            image/=255
        testImages[count] = image
        testResults[count][0] = 1   # idx 0 for empty
        count+=1

    if nbClasses>1:
        os.chdir(betweenImagesPath)
        for file in testBetweenImages:
            #print str(count+1) + " of " + str(totalLength)
            image = cv2.imread(betweenImagesPath+file, cv2.IMREAD_GRAYSCALE) ###GRAYSCALE!!!
            #image = np.float64(np.reshape(im, (imRows*imCols)))*1./255
            image = image.reshape(imRows, imCols, imChannels).astype("float32")
            if norm0to1:
                image/=255
            testImages[count] = image
            testResults[count][1] = 1   # 1 for between
            count+=1

    testData = (testImages, testResults)

    fullTestImages = []

    print "testData loaded"

    # Read FULL test images
    fullImages = sorted(os.listdir(fullImagesPath))
    for i in fullImages:
        if '.bmp' not in i:
            fullImages.remove(i)

    fullTestImages = np.zeros((len(fullImages), imRows, imCols, imChannels))
    os.chdir(fullImagesPath)
    count = 0
    for file in fullImages:
        #print str(count+1) + " of " + str(totalLength)
        image = cv2.imread(fullImagesPath+file, cv2.IMREAD_GRAYSCALE)
        image = image.reshape(imRows, imCols, imChannels).astype("float32")
        if norm0to1:
            image/=255
        fullTestImages[count] = image
        count+=1

    print "loaded all data."

    return (imRows, imCols, imChannels, trainData, valData, testData, fullTestImages)


def calcCrossEntropy(testImages, testPreds, nbBins=12):

    # testImages & testPreds should be of same size
    # testImages & testPreds must have values between 0&1

    imRows = testImages.shape[1]
    imCols = testImages.shape[2]
    imChannels = testImages.shape[3]

    rowTestImages = testImages.reshape(testImages.shape[0], imRows*imCols*imChannels)
    rowTestPreds = testPreds.reshape(testPreds.shape[0], imRows*imCols*imChannels)

    probRowTestImages = np.zeros((rowTestImages.shape[0], nbBins))
    for i in range(rowTestImages.shape[0]):
    	probRowTestImages[i] = np.bincount((rowTestImages[i]*10).astype(int), minlength=nbBins)/float(imRows*imCols*imChannels)

    probRowTestPreds = np.zeros((rowTestPreds.shape[0], nbBins))
    for i in range(rowTestPreds.shape[0]):
    	probRowTestPreds[i] = np.bincount((rowTestPreds[i]*10).astype(int), minlength=nbBins)/float(imRows*imCols*imChannels)

    testEntropies = np.zeros((probRowTestPreds.shape[0], 1))
    for i in range(probRowTestPreds.shape[0]):
    	nonZeroCols = np.nonzero(probRowTestPreds[i])
    	testEntropies[i] = (probRowTestImages[i][nonZeroCols]*np.log(probRowTestPreds[i][nonZeroCols]) + \
            (1-probRowTestImages[i][nonZeroCols])*np.log(1-probRowTestPreds[i][nonZeroCols])).sum()

    return testEntropies


# images is an ndarray of min. 2 dimensions
def resizeImages(images, imRows, imCols):

    print "resizing..."

    # image : rows x cols
    if (images.ndim==2):
        newImage = np.zeros((imRows, imCols))
        newImage = imresize(images, (imRows, imCols))/255.0
        return newImage

    # image : rows x cols x channels
    elif (images.ndim==3):
        newImages = np.zeros((imRows, imCols, images.shape[2]))
        for ch in range(images.shape[2]):
            newImage[:,:,ch] = imresize(images[:,:,ch], (imRows, imCols))/255.0

        print "ndims 3"
        return newImages

    # image : nOfImages x rows x cols x channels
    elif (images.ndim==4):
        newImages = np.zeros((images.shape[0], imRows, imCols, images.shape[3]))
        for i in range(images.shape[0]):
            for ch in range(images.shape[3]):
                newImages[i,:,:,ch] = imresize(images[i,:,:,ch], (imRows, imCols))/255.0

        return newImages

    else:
        print "image dimensions not 2, 3 or 4"
        return False


def KLT(images):

    flatImages = images.reshape(images.shape[0],images.shape[1]*images.shape[2]*images.shape[3])
    eigVals, eigVecs = np.linalg.eig(np.cov(flatImages))
    kltFlatImages = np.dot(eigVecs, flatImages)
    kltImages = kltFlatImages.reshape(images.shape[0], images.shape[1], images.shape[2], images.shape[3])
    return (kltImages, eigVecs, eigVals)
