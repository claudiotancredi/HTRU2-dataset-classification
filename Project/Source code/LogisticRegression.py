# -*- coding: utf-8 -*-
"""

@authors: Claudio Tancredi, Francesca Russo
"""

import numpy as np
import scipy.optimize
import LogRegFunctions

class LogisticRegression:

    def train(self, D, L, lambd, prior=0.5):
        self.x, self.f, self.d = scipy.optimize.fmin_l_bfgs_b(LogRegFunctions.logreg_obj, np.zeros(
            D.shape[0] + 1), args=(D, L, lambd, prior), approx_grad=True)
        
        

    def predict(self, X):
        scores = np.dot(self.x[0:-1], X) + self.x[-1]
        predictedLabels = (scores>0).astype(int)
        return predictedLabels
    
    def predictAndGetScores(self, X):
        scores = np.dot(self.x[0:-1], X) + self.x[-1]
        return scores
