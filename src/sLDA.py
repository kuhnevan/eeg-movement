# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:54:30 2019
@author: Ben
"""
from scipy.io import loadmat
#from scipy.io import savemat
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#left and right should be in form "[1, 2, 3, 4, 5, ...]"
left = "../../processed-data-image/s1-left-image.mat"
right = "../../processed-data-image/s1-right-image.mat"
leftMat = loadmat(left)
rightMat = loadmat(right)

X = np.array([leftMat, rightMat])
y = np.array([1, 2]) #1 corrosponds to left hand, 2 to right hand (could change)

for j in range(2, 53):
    left = "../../processed-data-image/s{}-left-image.mat".format(j)
    right = "../../processed-data-image/s{}-right-image.mat".format(j)
    leftMat = loadmat(left)
    rightMat = loadmat(right)
    a = np.array([leftMat, rightMat])
    
    #creates np array in the form [[1, 2, 3, ...], [1, 2, 3, ...], [1, 2, 3,...]]
    X = np.concatenate((X, a), axis=0)
    #creates np array in the form [1, 2, 1, 2, 1, 2, ...]
    y = np.concatenate((y, np.array([1, 2])), axis=None)
    
clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X, y)

#test prediction with testing
testImage = "../../processed-data-image/s1-left-image.mat"
testImageMat = loadmat(testImage)
print(clf.predict([testImageMat]))
