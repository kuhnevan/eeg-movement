# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 13:54:48 2019

@author: Kevin
"""
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np

for j in range(1, 53):
    left = loadmat("../../processed-data/s{}-left.mat".format(j))
    right = loadmat("../../processed-data/s{}-right.mat".format(j))
    
    leftMat = np.average(left.movement_left, axis=0)
    rightMat = np.average(right.movement_right, axis=0)
    
    savemat('../../processed-data/s{}-left-image.mat'.format(j),
            {'movement_left': leftMat})
    savemat('../../processed-data/s{}-right-image.mat'.format(j),
            {'movement_right': rightMat})