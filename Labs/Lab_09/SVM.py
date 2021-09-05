# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 15:28:01 2021

@author: Claudio
"""

import scipy.optimize as scopt
import numpy as np
import sklearn.datasets as skds
from itertools import repeat


def load_iris_binary():
    D, L = skds.load_iris()['data'].T, skds.load_iris()[
        'target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    # We assign label -1 (False) to virginica (was label 2), versicolor is 1 (True)
    L[L == 2] = -1
    return D, L


def split_db_2to1(D, L, seed=0):
    # Get an integer nTrain representing 2/3 of the dataset dimension
    nTrain = int(D.shape[1]*2.0/3.0)
    # Generate a random seed
    np.random.seed(seed)
    # D.shape[1] is 150, so an integer, so according to numpy documentation the
    # permutation function will work on the vector produced by arange(D.shape[1]),
    # which will be a 1-dim vector of 150 elements with evenly spaced values
    # between 0 and 149. Then these values are permuted.
    # The shuffle isn't really random, in fact with the same seed and the
    # same integer passed as parameter to the permutation function the
    # result will always be the same. This is because there's an algorithm
    # that outputs the values based on the seed and it obviously has a
    # deterministic behavior, so now we don't have to worry about getting
    # different values from the ones proposed in the pdf because they will
    # be the same.
    idx = np.random.permutation(D.shape[1])
    # In idxTrain we select only the first 100 elements of idx
    idxTrain = idx[0:nTrain]
    # In idxEval we select only the last 50 elements of idx
    idxEval = idx[nTrain:]
    # The Data matrix for TRaining need to be reduced to a 4x100 matrix.
    # At the same time, by passing the random vector idxTrain as a parameter
    # for the slice, we will actually select only the feature vectors (columns)
    # at indexes specified in the idxTrain vector.
    DTR = D[:, idxTrain]
    # The Data matrix for EValuation need to be reduced to a 4x50 matrix.
    # Same process of DTR. THERE CAN'T BE OVERLAPPING FEATURE VECTORS BETWEEN
    # DTR AND DEV, SINCE THE VALUES OF IDX THAT HAVE BEEN USED AS INDEXES ARE
    # ALL DIFFERENT.
    DEV = D[:, idxEval]
    # The Label vector for TRaining need to be reduced to size 100.
    # Same process, we pass an array with indexes.
    LTR = L[idxTrain]
    # The Label vector for EValuation need to be reduced to size 50.
    # Same process, we pass an array with indexes.
    LEV = L[idxEval]
    return (DTR, LTR), (DEV, LEV)


def LD_objectiveFunctionOfModifiedDualFormulation(alpha, H):
    grad = np.dot(H, alpha) - np.ones(H.shape[1])
    return ((1/2)*np.dot(np.dot(alpha.T, H), alpha)-np.dot(alpha.T, np.ones(H.shape[1])), grad)


def primalObjective(w, D, C, LTR, f):
    normTerm = (1/2)*(np.linalg.norm(w)**2)
    m = np.zeros(LTR.size)
    for i in range(LTR.size):
        vett = [0, 1-LTR[i]*(np.dot(w.T, D[:, i]))]
        m[i] = vett[np.argmax(vett)]
    pl = normTerm + C*np.sum(m)
    dl = -f
    dg = pl-dl
    return pl, dl, dg


def primalLossDualLossDualityGapErrorRate(DTR, C, Hij, LTR, LTE, DTE, D, K):
    b = list(repeat((0, C), DTR.shape[1]))
    (x, f, d) = scopt.fmin_l_bfgs_b(LD_objectiveFunctionOfModifiedDualFormulation,
                                    np.zeros(DTR.shape[1]), args=(Hij,), bounds=b, iprint=1, factr=1.0)
    # Now we can recover the primal solution
    w = np.sum((x*LTR).reshape(1, DTR.shape[1])*D, axis=1)
    # Compute the scores as in the previous lab
    S = np.dot(w.T, DTE)
    # Compute predicted labels. 1* is useful to convert True/False to 1/0
    LP = 1*(S > 0)
    # Replace 0 with -1 because of the transformation that we did on the labels
    LP[LP == 0] = -1
    numberOfCorrectPredictions = np.array(LP == LTE).sum()
    accuracy = numberOfCorrectPredictions/LTE.size*100
    errorRate = 100-accuracy
    # Compute primal loss, dual loss, duality gap
    pl, dl, dg = primalObjective(w, D, C, LTR, f)
    print("K=%d, C=%f, Primal loss=%e, Dual loss=%e, Duality gap=%e, Error rate=%.1f %%" % (
        K, C, pl, dl, dg, errorRate))
    return


def modifiedDualFormulation(DTR, LTR, DTE, LTE, K):
    # Compute the D matrix for the extended training set with K=1
    row = np.zeros(DTR.shape[1])+K
    D = np.vstack([DTR, row])
    row = np.zeros(DTE.shape[1])+K
    DTE = np.vstack([DTE, row])
    # Compute the H matrix exploiting broadcasting
    Gij = np.dot(D.T, D)
    # To compute zi*zj I need to reshape LTR as a matrix with one column/row
    # and then do the dot product
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*Gij
    # We want to maximize JD(alpha), but we can't use the same algorithm of the
    # previous lab, so we can cast the problem as minimization of LD(alpha) defined
    # as -JD(alpha)
    # We use three different values for hyperparameter C
    # 1) C=0.1
    primalLossDualLossDualityGapErrorRate(DTR, 0.1, Hij, LTR, LTE, DTE, D, K)
    # 2) C=1
    primalLossDualLossDualityGapErrorRate(DTR, 1, Hij, LTR, LTE, DTE, D, K)
    # 3) C=10
    primalLossDualLossDualityGapErrorRate(DTR, 10, Hij, LTR, LTE, DTE, D, K)
    return


def dualLossErrorRatePoly(DTR, C, Hij, LTR, LTE, DTE, K, d, c):
    b = list(repeat((0, C), DTR.shape[1]))
    (x, f, data) = scopt.fmin_l_bfgs_b(LD_objectiveFunctionOfModifiedDualFormulation,
                                    np.zeros(DTR.shape[1]), args=(Hij,), bounds=b, iprint=1, factr=1.0)
    # Compute the scores
    S = np.sum(
        np.dot((x*LTR).reshape(1, DTR.shape[1]), (np.dot(DTR.T, DTE)+c)**d+ K), axis=0)
    # Compute predicted labels. 1* is useful to convert True/False to 1/0
    LP = 1*(S > 0)
    # Replace 0 with -1 because of the transformation that we did on the labels
    LP[LP == 0] = -1
    numberOfCorrectPredictions = np.array(LP == LTE).sum()
    accuracy = numberOfCorrectPredictions/LTE.size*100
    errorRate = 100-accuracy
    # Compute dual loss
    dl = -f
    print("K=%d, C=%f, Kernel Poly (d=%d, c=%d), Dual loss=%e, Error rate=%.1f %%" % (K, C, d, c, dl, errorRate))
    return

def dualLossErrorRateRBF(DTR, C, Hij, LTR, LTE, DTE, K, gamma):
    b = list(repeat((0, C), DTR.shape[1]))
    (x, f, data) = scopt.fmin_l_bfgs_b(LD_objectiveFunctionOfModifiedDualFormulation,
                                    np.zeros(DTR.shape[1]), args=(Hij,), bounds=b, iprint=1, factr=1.0)
    kernelFunction = np.zeros((DTR.shape[1], DTE.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTE.shape[1]):
            kernelFunction[i,j]=RBF(DTR[:, i], DTE[:, j], gamma, K)
    S=np.sum(np.dot((x*LTR).reshape(1, DTR.shape[1]), kernelFunction), axis=0)
    # Compute the scores
    # S = np.sum(
    #     np.dot((x*LTR).reshape(1, DTR.shape[1]), (np.dot(DTR.T, DTE)+c)**d+ K), axis=0)
    # Compute predicted labels. 1* is useful to convert True/False to 1/0
    LP = 1*(S > 0)
    # Replace 0 with -1 because of the transformation that we did on the labels
    LP[LP == 0] = -1
    numberOfCorrectPredictions = np.array(LP == LTE).sum()
    accuracy = numberOfCorrectPredictions/LTE.size*100
    errorRate = 100-accuracy
    # Compute dual loss
    dl = -f
    print("K=%d, C=%f, RBF (gamma=%d), Dual loss=%e, Error rate=%.1f %%" % (K, C, gamma, dl, errorRate))
    return


def kernelPoly(DTR, LTR, DTE, LTE, K, C, d, c):
    # Compute the H matrix exploiting broadcasting
    kernelFunction = (np.dot(DTR.T, DTR)+c)**d+ K**2
    # To compute zi*zj I need to reshape LTR as a matrix with one column/row
    # and then do the dot product
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*kernelFunction
    # We want to maximize JD(alpha), but we can't use the same algorithm of the
    # previous lab, so we can cast the problem as minimization of LD(alpha) defined
    # as -JD(alpha)
    dualLossErrorRatePoly(DTR, C, Hij, LTR, LTE, DTE, K, d, c)
    return


def RBF(x1, x2, gamma, K):
    return np.exp(-gamma*(np.linalg.norm(x1-x2)**2))+K**2

def kernelRBF(DTR, LTR, DTE, LTE, K, C, gamma):
    # Compute the H matrix exploiting broadcasting
    kernelFunction = np.zeros((DTR.shape[1], DTR.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            kernelFunction[i,j]=RBF(DTR[:, i], DTR[:, j], gamma, K)
    # To compute zi*zj I need to reshape LTR as a matrix with one column/row
    # and then do the dot product
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*kernelFunction
    # We want to maximize JD(alpha), but we can't use the same algorithm of the
    # previous lab, so we can cast the problem as minimization of LD(alpha) defined
    # as -JD(alpha)
    dualLossErrorRateRBF(DTR, C, Hij, LTR, LTE, DTE, K, gamma)
    return


if __name__ == "__main__":
    # ---------------- LINEAR SVM ---------------------
    # We'll discriminate between iris virginica and iris versicolor. We will
    # ignore iris setosa. We will represent labels with 1 (iris versicolor) and
    # 0 (iris virginica).
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    # K=1
    modifiedDualFormulation(DTR, LTR, DTE, LTE, 1)
    # K=10
    modifiedDualFormulation(DTR, LTR, DTE, LTE, 10)
    # ---------------- KERNEL SVM ----------------------
    # The parameters 0,1,2,0 are K, C, d, c
    kernelPoly(DTR, LTR, DTE, LTE, 0, 1, 2, 0)
    kernelPoly(DTR, LTR, DTE, LTE, 1, 1, 2, 0)
    kernelPoly(DTR, LTR, DTE, LTE, 0, 1, 2, 1)
    kernelPoly(DTR, LTR, DTE, LTE, 1, 1, 2, 1)
    # The parameters 0,1,1 are K, C, gamma
    kernelRBF(DTR, LTR, DTE, LTE, 0, 1, 1)
    kernelRBF(DTR, LTR, DTE, LTE, 0, 1, 10)
    kernelRBF(DTR, LTR, DTE, LTE, 1, 1, 1)
    kernelRBF(DTR, LTR, DTE, LTE, 1, 1, 10)
