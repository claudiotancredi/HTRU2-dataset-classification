# -*- coding: utf-8 -*-
"""

@authors: Claudio Tancredi, Francesca Russo
"""

import numpy as np
import utils
import LBGalgorithm
import multivariateGaussianGMM


class GMMTiedCov():
    def train (self, D, L, components):
        D0 = D[:, L==0]
        D1 = D[:, L==1]
       
        sigma0 =  np.cov(D0).reshape((D0.shape[0], D0.shape[0]))
        sigma1 =  np.cov(D1).reshape((D1.shape[0], D1.shape[0]))
        
        self.sigma = 1/(D.shape[1])*(D[:, L == 0].shape[1]*sigma0+D[:, L == 1].shape[1]*sigma1)
        GMM0_init = [(1.0, D0.mean(axis=1).reshape((D0.shape[0], 1)), utils.constrainSigma(self.sigma))]
        GMM1_init = [(1.0, D1.mean(axis=1).reshape((D1.shape[0], 1)), utils.constrainSigma(self.sigma))]
       

        self.GMM0 = LBGalgorithm.TiedLBGalgorithm (GMM0_init, D0, components)
        self.GMM1 = LBGalgorithm.TiedLBGalgorithm (GMM1_init, D1, components)

         
        
        
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
        
        self.GMM0 = LBGalgorithm.TiedLBGalgorithm (GMM0, D0, 1)
        self.GMM1 = LBGalgorithm.TiedLBGalgorithm (GMM1, D1, 1)
        
        return [self.GMM0, self.GMM1]