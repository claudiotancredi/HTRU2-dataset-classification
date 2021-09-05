# -*- coding: utf-8 -*-
"""

@authors: Claudio Trancredi, Francesca Russo
"""

import utils
import numpy as np
import multivariateGaussian
import numpy.matlib 



class GaussianClassifierNB:
    
    def train (self, D, L):
         self.mean0 = utils.mcol(D[:, L == 0].mean(axis=1))
         self.mean1 = utils.mcol(D[:, L == 1].mean(axis=1))
        
         
         
        
         I=np.matlib.identity(D.shape[0])
         self.sigma0 =   np.multiply(np.cov(D[:, L == 0]),I)
         self.sigma1 =   np.multiply(np.cov(D[:, L == 1]),I)
        
 
         
         #class priors
         self.pi0 = D[:, L==0].shape[1]/D.shape[1]
         self.pi1 = D[:, L==1].shape[1]/D.shape[1]
       

       
         
     
    def predict (self, X):
        LS0 = np.asarray(multivariateGaussian.logpdf_GAU_ND(X, self.mean0, self.sigma0 )).flatten()
        LS1 = np.asarray(multivariateGaussian.logpdf_GAU_ND(X, self.mean1, self.sigma1 )).flatten()
      
        LS = np.vstack((LS0, LS1))
        
        #Log SJoints, that is the joint log-probabilities for a given sample
        LSJoint =  multivariateGaussian.joint_log_density(LS, utils.mcol(np.array([np.log(self.pi0), np.log(self.pi1) ])))
        
        #marginal log densities
        MLD = multivariateGaussian.marginal_log_densities(LSJoint)
        
        #Log-posteriors
        LP = multivariateGaussian.log_posteriors(LSJoint, MLD)
       
        
        return  np.argmax(LP, axis=0)
    
    def predictAndGetScores (self, X):
        LS0 = np.asarray(multivariateGaussian.logpdf_GAU_ND(X, self.mean0, self.sigma0 )).flatten()
        LS1 = np.asarray(multivariateGaussian.logpdf_GAU_ND(X, self.mean1, self.sigma1 )).flatten()
        
        #log-likelihood ratios
        llr = LS1-LS0
        return llr
    
