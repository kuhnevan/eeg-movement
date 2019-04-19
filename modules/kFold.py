# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:25:11 2019

@author: Ben
"""
from scipy.io import loadmat
#from scipy.io import savemat
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lowRange = 1
highRange = 53

numTested = 0;
numRight = 0;
numRH = 0;
numLH = 0

for i in range(lowRange, highRange):
    starterLoc = i+1 if i+1 < highRange else 1
    
    left = "../../processed-data-image/s{}-left-image.mat".format(starterLoc)
    right = "../../processed-data-image/s{}-right-image.mat".format(starterLoc)
    leftMat = loadmat(left)["movement_left"]
    rightMat = loadmat(right)["movement_right"]
    
    X = np.concatenate((leftMat, rightMat), axis=0)

    yfiller = np.full((1, leftMat.shape[0]), 1, dtype=int)[0]
    y = yfiller
    yfiller = np.full((1, rightMat.shape[0]), 2, dtype=int)[0]
    y = np.concatenate((y, yfiller), axis=None)
    
    for ii in range(lowRange, highRange):
        if ii==i or ii==starterLoc: 
            continue
        left = "../../processed-data-image/s{}-left-image.mat".format(ii)
        right = "../../processed-data-image/s{}-right-image.mat".format(ii)
        leftMat = loadmat(left)["movement_left"]
        rightMat = loadmat(right)["movement_right"]
        
        #creates np array in the form [[1, 2, 3, ...], [1, 2, 3, ...], [1, 2, 3,...]]
        X = np.concatenate((X, leftMat), axis=0)
        X = np.concatenate((X, rightMat), axis=0)
        #creates np array in the form [1, 2, 1, 2, 1, 2, ...]
        
        yfiller = np.full((1, leftMat.shape[0]), 1, dtype=int)[0]
        y = np.concatenate((y, yfiller), axis=None)
        yfiller = np.full((1, rightMat.shape[0]), 2, dtype=int)[0]
        y = np.concatenate((y, yfiller), axis=None)
    
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X, y)
    
    testImage = "../../processed-data-image/s{}-left-image.mat".format(i)
    testImageMat = loadmat(testImage)["movement_left"]
    resultMat = clf.predict(testImageMat)
    for aResult in resultMat:
        numTested = numTested + 1
        numLH = numLH + 1
        if aResult==1:
            numRight = numRight + 1
    
    testImage = "../../processed-data-image/s{}-right-image.mat".format(i)
    testImageMat = loadmat(testImage)["movement_right"]
    resultMat = clf.predict(testImageMat)
    for aResult in resultMat:
        numTested = numTested + 1
        numRH = numRH + 1
        if aResult==2:
            numRight = numRight + 1
    
print(numRight/numTested)
print(numRH)
print(numLH)        
