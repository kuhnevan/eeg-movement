# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:54:30 2019
@author: Ben
"""
from scipy.io import loadmat
#from scipy.io import savemat
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot


def runSLDA(fileloc):
    # left and right should be in form "[[1, 2, 3, ...], [3, 4, 6, ...], [...], ...]"
    
    left = fileloc + "processed-data-image/s1-left-image.mat"
    right = fileloc + "processed-data-image/s1-right-image.mat"
    leftMat = loadmat(left)["movement_left"]
    rightMat = loadmat(right)["movement_right"]
    
    trainX = []
    trainY = []
    
    for j in range(1, 37):
        left = fileloc + "processed-data-image/s{}-left-image.mat".format(j)
        right = fileloc + "processed-data-image/s{}-right-image.mat".format(j)
        leftMat = loadmat(left)["movement_left"]
        rightMat = loadmat(right)["movement_right"]
        
        #creates np array in the form [[1, 2, 3, ...], [1, 2, 3, ...], [1, 2, 3,...]]
        trainX.append(leftMat)
        trainX.append(rightMat)
        #creates np array in the form [1, 2, 1, 2, 1, 2, ...]
        
        trainY.append(np.full(leftMat.shape[0], 1))
        trainY.append(np.full(rightMat.shape[0], 0))
        
    trainX = np.array(trainX)
    trainX = np.reshape(trainX, (1368, 69))
    trainY = np.array(trainY)
    trainY = trainY.flatten()
        
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(trainX, trainY)
    
    #test prediction with testing
    #testImage = "../desktop/448/processed-data-image/s1-left-image.mat"
    #testImageMat = loadmat(testImage)["movement_left"]
    #print(clf.predict(testImageMat))
    
    numTested = 0;
    numRight = 0;
    numRH = 0;
    numLH = 0
    
    testX = []
    testY = []
    
    for k in range(37, 53):
        testImage = fileloc + "processed-data-image/s{}-left-image.mat".format(k)
        testImageMat = loadmat(testImage)["movement_left"]
        testX.append(testImageMat)
        testY.append(np.full(testImageMat.shape[0], 1))
        resultMat = clf.predict(testImageMat)
        for aResult in resultMat:
            numTested = numTested + 1
            numLH = numLH + 1
            if aResult==1:
                numRight = numRight + 1
        
        testImage = fileloc + "processed-data-image/s{}-right-image.mat".format(k)
        testImageMat = loadmat(testImage)["movement_right"]
        testX.append(testImageMat)
        testY.append(np.full(testImageMat.shape[0], 0))
        resultMat = clf.predict(testImageMat)
        for aResult in resultMat:
            numTested = numTested + 1
            numRH = numRH + 1
            if aResult==0:
                numRight = numRight + 1
    
    testX = np.array(testX)
    testX = np.reshape(testX, (608, 69))
    testY = np.array(testY)
    testY = testY.flatten()
    
    y_true = testY
    probs = clf.predict_proba(testX)
    #  keep probabilities for positive outcome only
    probs = probs[:, 1]
    y_scores = probs
    score  = roc_auc_score(y_true, y_scores)
    
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(testY, probs)
    # plot no skill
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    pyplot.plot(fpr, tpr, marker='.')
    # plot labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('Receiver Operating Characteristic')
    # show the plot
    #pyplot.show()
    pyplot.savefig(fileloc + "aucFig.png")
                
    print(numRight/numTested)
    print(numRH)
    print(numLH)
    print(score)
    return score;