# -*- coding: utf-8 -*-
"""


@authors: Claudio Tancredi, Francesca Russo
"""
import numpy as np
import utils
import LBGalgorithm
import multivariateGaussianGMM


class GMMDiag():
    def train (self, D, L, components):
        D0 = D[:, L==0]
        D1 = D[:, L==1]
       
       
        GMM0_init = [(1.0, D0.mean(axis=1).reshape((D0.shape[0], 1)), utils.constrainSigma(np.cov(D0)*np.eye( D0.shape[0]).reshape((D0.shape[0]),D0.shape[0])))]
        GMM1_init = [(1.0, D1.mean(axis=1).reshape((D1.shape[0], 1)), utils.constrainSigma(np.cov(D1)*np.eye( D1.shape[0]).reshape((D1.shape[0]),D1.shape[0])))]
       

        self.GMM0 = LBGalgorithm.DiagLBGalgorithm (GMM0_init, D0, components)
        self.GMM1 = LBGalgorithm.DiagLBGalgorithm (GMM1_init, D1, components)
     
        
        
    def predict (self, X):

        
        PD0 = multivariateGaussianGMM.compute_posterior_GMM(X, self.GMM0)
        PD1 = multivariateGaussianGMM.compute_posterior_GMM(X, self.GMM1)
      
 
        PD = np.vstack((PD0,PD1))
        return np.argmax(PD, axis=0)
    
    def predictAndGetScores(self, X):
    
        LS0 = multivariateGaussianGMM.computeLogLikelihood(X, self.GMM0)
        LS1 = multivariateGaussianGMM.computeLogLikelihood(X, self.GMM1)
        
        llr = LS1-LS0
        return llr

    def fastTraining(self, D, L, GMM0, GMM1):
        D0 = D[:, L==0]
        D1 = D[:, L==1]
        
        self.GMM0 = LBGalgorithm.DiagLBGalgorithm (GMM0, D0, 1)
        self.GMM1 = LBGalgorithm.DiagLBGalgorithm (GMM1, D1, 1)
        
        return [self.GMM0, self.GMM1]