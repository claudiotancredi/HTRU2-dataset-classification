# -*- coding: utf-8 -*-
"""

@authors: Claudio Tancredi, Francesca Russo
"""

import scipy.optimize 
import numpy as np
from itertools import repeat

def LD_objectiveFunctionOfModifiedDualFormulation(alpha, H):
    grad = np.dot(H, alpha) - np.ones(H.shape[1])
    return ((1/2)*np.dot(np.dot(alpha.T, H), alpha)-np.dot(alpha.T, np.ones(H.shape[1])), grad)

def modifiedDualFormulation(DTR, LTR, C, K):
    # Compute the D matrix for the extended training set
    
    row = np.zeros(DTR.shape[1])+K
    D = np.vstack([DTR, row])

    # Compute the H matrix 
    Gij = np.dot(D.T, D)
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*Gij
    
    b = list(repeat((0, C), DTR.shape[1]))
    (x, f, d) = scipy.optimize .fmin_l_bfgs_b(LD_objectiveFunctionOfModifiedDualFormulation,
                                    np.zeros(DTR.shape[1]), args=(Hij,), bounds=b, iprint=1, factr=1.0)
    return np.sum((x*LTR).reshape(1, DTR.shape[1])*D, axis=1)



def kernelPoly(DTR, LTR, K, C, d, c):
    # Compute the H matrix
    kernelFunction = (np.dot(DTR.T, DTR)+c)**d+ K**2
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*kernelFunction
    b = list(repeat((0, C), DTR.shape[1]))
    (x, f, data) = scipy.optimize.fmin_l_bfgs_b(LD_objectiveFunctionOfModifiedDualFormulation,
                                    np.zeros(DTR.shape[1]), args=(Hij,), bounds=b, iprint=1, factr=1.0)
    return x


def RBF(x1, x2, gamma, K):
    return np.exp(-gamma*(np.linalg.norm(x1-x2)**2))+K**2

def kernelRBF(DTR, LTR, gamma,  K, C):
    # Compute the H matrix
    kernelFunction = np.zeros((DTR.shape[1], DTR.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            kernelFunction[i,j]=RBF(DTR[:, i], DTR[:, j], gamma, K)
    zizj = np.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    Hij = zizj*kernelFunction
    b = list(repeat((0, C), DTR.shape[1]))
    (x, f, data) = scipy.optimize.fmin_l_bfgs_b(LD_objectiveFunctionOfModifiedDualFormulation,
                                    np.zeros(DTR.shape[1]), args=(Hij,), bounds=b, iprint=1, factr=1.0)
    return x


class SVM ():
    
    def train (self, DTR, LTR, option, c=0, d=2, gamma=1.0, C=1.0, K=1.0 ):
        self.option = option
        self.DTR = DTR
        self.LTR = LTR
        self.K = K
        self.C = C
        if (option == 'linear'):
            self.w = modifiedDualFormulation(DTR, LTR, self.C, self.K)
            
        if (option == 'polynomial'):
            self.c = c
            self.d = d
            self.x = kernelPoly(DTR, LTR, K, C, d, c)
            
        if (option == 'RBF'):
            self.gamma = gamma
            self.x = kernelRBF(DTR, LTR, gamma, K, C) 
            
    
    def predict (self, DTE):
        
        if (self.option == 'linear'):

             DTE = np.vstack([DTE, np.zeros(DTE.shape[1])+self.K])
             S = np.dot(self.w.T, DTE)
             LP = 1*(S > 0)
             LP[LP == 0] = -1
             return LP
        if (self.option == 'polynomial'):

            S = np.sum(
                np.dot((self.x*self.LTR).reshape(1, self.DTR.shape[1]), (np.dot(self.DTR.T, DTE)+self.c)**self.d+ self.K), axis=0)
            LP = 1*(S > 0)
            LP[LP == 0] = -1
            return LP
        if (self.option == 'RBF'):
            
            kernelFunction = np.zeros((self.DTR.shape[1], DTE.shape[1]))
            for i in range(self.DTR.shape[1]):
                for j in range(DTE.shape[1]):
                    kernelFunction[i,j]=RBF(self.DTR[:, i], DTE[:, j], self.gamma, self.K)
            S=np.sum(np.dot((self.x*self.LTR).reshape(1, self.DTR.shape[1]), kernelFunction), axis=0)
            LP = 1*(S > 0)
            LP[LP == 0] = -1  
            return LP
    
    def predictAndGetScores(self, DTE):
        if (self.option == 'linear'):

             DTE = np.vstack([DTE, np.zeros(DTE.shape[1])+self.K])
             S = np.dot(self.w.T, DTE)
             return S
        if (self.option == 'polynomial'):

            S = np.sum(
                np.dot((self.x*self.LTR).reshape(1, self.DTR.shape[1]), (np.dot(self.DTR.T, DTE)+self.c)**self.d+ self.K), axis=0)
            return S
        if (self.option == 'RBF'):
            
            kernelFunction = np.zeros((self.DTR.shape[1], DTE.shape[1]))
            for i in range(self.DTR.shape[1]):
                for j in range(DTE.shape[1]):
                    kernelFunction[i,j]=RBF(self.DTR[:, i], DTE[:, j], self.gamma, self.K)
            S=np.sum(np.dot((self.x*self.LTR).reshape(1, self.DTR.shape[1]), kernelFunction), axis=0)
            return S
    