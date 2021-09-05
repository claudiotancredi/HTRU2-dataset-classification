# -*- coding: utf-8 -*-
"""

@author: Claudio Tancredi, Francesca Russo
"""

import utils
import numpy as np
import multivariateGaussian


class GaussianClassifier:
    
    def train (self, D, L):
         self.mean0 = utils.mcol(D[:, L == 0].mean(axis=1))
         self.mean1 = utils.mcol(D[:, L == 1].mean(axis=1))
        
         
         self.sigma0 = np.cov(D[:, L == 0])
         self.sigma1 = np.cov(D[:, L == 1])
         
         
         #class priors
         self.pi0 = D[:, L==0].shape[1]/D.shape[1]
         self.pi1 = D[:, L==1].shape[1]/D.shape[1]
       
         
     
    def predict (self, X):
        LS0 = multivariateGaussian.logpdf_GAU_ND(X, self.mean0, self.sigma0 )
        LS1 = multivariateGaussian.logpdf_GAU_ND(X, self.mean1, self.sigma1 )
        
        LS = np.vstack((LS0, LS1))
        
        #Log SJoints, that is the joint log-probabilities for a given sample
        LSJoint =  multivariateGaussian.joint_log_density(LS, utils.mcol(np.array([np.log(self.pi0), np.log(self.pi1) ])))
        
        #marginal log densities
        MLD = multivariateGaussian.marginal_log_densities(LSJoint)
        
        #Log-posteriors
        LP = multivariateGaussian.log_posteriors(LSJoint, MLD)
        
       
        predictions = np.argmax(LP, axis=0)
        
        return  predictions
    
    def predictAndGetScores (self, X):
        LS0 = multivariateGaussian.logpdf_GAU_ND(X, self.mean0, self.sigma0 )
        LS1 = multivariateGaussian.logpdf_GAU_ND(X, self.mean1, self.sigma1 )
        #log-likelihood ratios
        llr = LS1-LS0
        return llr
