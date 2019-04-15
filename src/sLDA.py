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

# left and right should be in form "[[1, 2, 3, ...], [3, 4, 6, ...], [...], ...]"

left = "../../processed-data-image/s1-left-image.mat"
right = "../../processed-data-image/s1-right-image.mat"
leftMat = loadmat(left)["movement_left"]
rightMat = loadmat(right)["movement_right"]

X = np.concatenate((leftMat, rightMat), axis=0)

yfiller = np.full((1, leftMat.shape[0]), 1, dtype=int)[0]
y = yfiller
yfiller = np.full((1, rightMat.shape[0]), 0, dtype=int)[0]
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
    yfiller = np.full((1, rightMat.shape[0]), 0, dtype=int)[0]
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

testX = []
testY = []



for k in range(37, 53):
    testImage = "../../processed-data-image/s{}-left-image.mat".format(k)
    testImageMat = loadmat(testImage)["movement_left"]
    testX.append(testImageMat)
    testY.append(np.full(testImageMat.shape[0], 1))
    resultMat = clf.predict(testImageMat)
    for aResult in resultMat:
        numTested = numTested + 1
        numLH = numLH + 1
        if aResult==1:
            numRight = numRight + 1
    
    testImage = "../../processed-data-image/s{}-right-image.mat".format(k)
    testImageMat = loadmat(testImage)["movement_right"]
    testX.append(testImageMat)
    testY.append(np.full(testImageMat.shape[0], 0))
    resultMat = clf.predict(testImageMat)
    for aResult in resultMat:
        numTested = numTested + 1
        numRH = numRH + 1
        if aResult==2:
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
pyplot.show()
            
print(numRight/numTested)
print(numRH)
print(numLH)
print(score)
