# -*- coding: utf-8 -*-
"""

@authors: Claudio Tancredi, Francesca Russo
"""
import EMalgorithm
import numpy as np

def split(GMM, alpha = 0.1):
    
    size = len(GMM)
    splittedGMM = []
    for i in range(size):
        U, s, Vh = np.linalg.svd(GMM[i][2])
        # compute displacement vector
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        splittedGMM.append((GMM[i][0]/2, GMM[i][1]+d, GMM[i][2]))
        splittedGMM.append((GMM[i][0]/2, GMM[i][1]-d, GMM[i][2]))
    return splittedGMM

def LBGalgorithm(GMM, X, iterations):
    GMM = EMalgorithm.EMalgorithm(X, GMM)
    for i in range(iterations):
        GMM = split(GMM)
        GMM = EMalgorithm.EMalgorithm(X, GMM)
    return GMM

def DiagLBGalgorithm(GMM, X, iterations):
    GMM = EMalgorithm.DiagEMalgorithm(X, GMM)
    for i in range(iterations):
        GMM = split(GMM)
        GMM = EMalgorithm.DiagEMalgorithm(X, GMM)
    return GMM


def TiedLBGalgorithm(GMM, X, iterations):
    GMM = EMalgorithm.TiedEMalgorithm(X, GMM)
    for i in range(iterations):
        GMM = split(GMM)
        GMM = EMalgorithm.TiedEMalgorithm(X, GMM)
    return GMM
