# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:54:30 2019
@author: Ben
"""
from scipy.io import loadmat
#from scipy.io import savemat
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# left and right should be in form "[[1, 2, 3, ...], [3, 4, 6, ...], [...], ...]"

left = "../../processed-data-image/s1-left-image.mat"
right = "../../processed-data-image/s1-right-image.mat"
leftMat = loadmat(left)["movement_left"]
rightMat = loadmat(right)["movement_right"]

X = np.concatenate((leftMat, rightMat), axis=0)

yfiller = np.full((1, leftMat.shape[0]), 1, dtype=int)[0]
y = yfiller
yfiller = np.full((1, rightMat.shape[0]), 2, dtype=int)[0]
y = np.concatenate((y, yfiller), axis=None)



for j in range(2, 37):
    left = "../../processed-data-image/s{}-left-image.mat".format(j)
    right = "../../processed-data-image/s{}-right-image.mat".format(j)
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

#test prediction with testing
#testImage = "../desktop/448/processed-data-image/s1-left-image.mat"
#testImageMat = loadmat(testImage)["movement_left"]
#print(clf.predict(testImageMat))

numTested = 0;
numRight = 0;
numRH = 0;
numLH = 0

for k in range(37, 53):
    testImage = "../../processed-data-image/s{}-left-image.mat".format(k)
    testImageMat = loadmat(testImage)["movement_left"]
    resultMat = clf.predict(testImageMat)
    for aResult in resultMat:
        numTested = numTested + 1
        numLH = numLH + 1
        if aResult==1:
            numRight = numRight + 1
    
    testImage = "../../processed-data-image/s{}-right-image.mat".format(k)
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
