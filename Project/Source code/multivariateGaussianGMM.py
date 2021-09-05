# -*- coding: utf-8 -*-
"""

@authors: Claudio Tancredi, Francesca Russo
"""

import numpy as np
import scipy.special 
import multivariateGaussian


def logpdf_GMM(X, gmm):
    S = np.zeros((len(gmm), X.shape[1]))
    for i in range(len(gmm)):
        # Compute log density
        S[i, :] = multivariateGaussian.logpdf_GAU_ND(X, gmm[i][1], gmm[i][2])
    return S

def joint_log_density_GMM (S, gmm):
    
    for i in range(len(gmm)):
        # Add log of the prior of the corresponding component
        S[i, :] += np.log(gmm[i][0])
    return S

def marginal_density_GMM (S):
    return scipy.special.logsumexp(S, axis = 0)


def log_likelihood_GMM(logmarg, X):
    return np.sum(logmarg)/X.shape[1]

def compute_posterior_GMM(X, gmm):
     return marginal_density_GMM(joint_log_density_GMM(logpdf_GMM(X, gmm),gmm))
 
def computeLogLikelihood(X, gmm):
    # SHOULD BE FIXED
    tempSum=np.zeros((len(gmm), X.shape[1]))
    for i in range(len(gmm)):
        tempSum[i,:]=np.log(gmm[i][0])+multivariateGaussian.logpdf_GAU_ND(X, gmm[i][1], gmm[i][2])
    return scipy.special.logsumexp(tempSum, axis=0)