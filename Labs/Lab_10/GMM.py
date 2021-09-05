# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 10:06:54 2021

@author: Claudio
"""

import numpy as np
from GMM_load import load_gmm
import scipy.special as scsp
import matplotlib.pyplot as plt
import sklearn.datasets as skds


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


def load_iris():
    # Here is shown another way to load the iris dataset. We can load it from
    # the sklearn library but we need to transpose the data matrix, since we
    # work with a column representation of feature vectors
    D, L = skds.load_iris()['data'].T, skds.load_iris()['target']
    return D, L


def mcol(v):
    return v.reshape((v.size, 1))


def logpdf_GAU_ND(x, mu, sigma):
    # There are two options here, read README.md for an explanation
    # return np.diag(-(x.shape[0]/2)*np.log(2*np.pi)-(1/2)*(np.linalg.slogdet(sigma)[1])-(1/2)*np.dot(np.dot((x-mu).T,np.linalg.inv(sigma)),(x-mu)))
    # or
    return -(x.shape[0]/2)*np.log(2*np.pi)-(1/2)*(np.linalg.slogdet(sigma)[1])-(1/2)*((np.dot((x-mu).T, np.linalg.inv(sigma))).T*(x-mu)).sum(axis=0)


def logpdf_GMM(X, gmm):
    # This function will compute the log-density of a GMM for a set of samples
    # contained in matrix X of shape (D, N), where D is the size of a sample
    # and N is the number of samples in X.
    # We define a matrix S with shape (M, N). Each row will contain the sub-class
    # conditional densities given component Gi for all samples xi
    S = np.zeros((len(gmm), X.shape[1]))
    for i in range(len(gmm)):
        # Compute log density
        S[i, :] = logpdf_GAU_ND(X, gmm[i][1], gmm[i][2])
        # Add log of the prior of the corresponding component
        S[i, :] += np.log(gmm[i][0])
    # Compute the log-marginal log fxi(xi). The result will be an array of shape
    # (N,) whose component i will contain the log-density for sample xi
    logdens = scsp.logsumexp(S, axis=0)
    return (logdens, S)


def Estep(logdens, S):
    # E-step: compute the POSTERIOR PROBABILITY (=responsibilities) for each component of the GMM
    # for each sample, using the previous estimate of the model parameters.
    return np.exp(S-logdens.reshape(1, logdens.size))


def constrainSigma(sigma):
    psi = 0.01
    U, s, Vh = np.linalg.svd(sigma)
    s[s < psi] = psi
    sigma = np.dot(U, mcol(s)*U.T)
    return sigma


def DiagConstrainSigma(sigma):
    sigma = sigma * np.eye(sigma.shape[0])
    psi = 0.01
    U, s, Vh = np.linalg.svd(sigma)
    s[s < psi] = psi
    sigma = np.dot(U, mcol(s)*U.T)
    return sigma


def Mstep(X, S, posterior):
    psi = 0.01
    # M-step: update the model parameters.
    Zg = np.sum(posterior, axis=1)  # 3
    # print(Zg)
    # Fg = np.array([np.sum(posterior[0, :].reshape(1, posterior.shape[1])* X, axis=1), np.sum(posterior[1, :].reshape(1, posterior.shape[1])* X, axis=1), np.sum(posterior[2, :].reshape(1, posterior.shape[1])*X, axis=1)])
    # print(Fg)
    Fg = np.zeros((X.shape[0], S.shape[0]))  # 4x3
    for g in range(S.shape[0]):
        tempSum = np.zeros(X.shape[0])
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * X[:, i]
        Fg[:, g] = tempSum
    # print(Fg)
    Sg = np.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        tempSum = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * np.dot(X[:, i].reshape(
                (X.shape[0], 1)), X[:, i].reshape((1, X.shape[0])))
        Sg[g] = tempSum
    # print(Sg)
    mu = Fg / Zg
    prodmu = np.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        prodmu[g] = np.dot(mu[:, g].reshape((X.shape[0], 1)),
                           mu[:, g].reshape((1, X.shape[0])))
    # print(prodmu)
    # print(np.dot(mu, mu.T).reshape((1, mu.shape[0], mu.shape[0]))) NO, it is wrong
    cov = Sg / Zg.reshape((Zg.size, 1, 1)) - prodmu
    # The following two lines of code are used to model the constraints
    for g in range(S.shape[0]):
        U, s, Vh = np.linalg.svd(cov[g])
        s[s < psi] = psi
        cov[g] = np.dot(U, mcol(s)*U.T)
    # print(cov)
    w = Zg/np.sum(Zg)
    # print(w)
    return (w, mu, cov)


def DiagMstep(X, S, posterior):
    psi = 0.01
    # M-step: update the model parameters.
    Zg = np.sum(posterior, axis=1)  # 3
    # print(Zg)
    # Fg = np.array([np.sum(posterior[0, :].reshape(1, posterior.shape[1])* X, axis=1), np.sum(posterior[1, :].reshape(1, posterior.shape[1])* X, axis=1), np.sum(posterior[2, :].reshape(1, posterior.shape[1])*X, axis=1)])
    # print(Fg)
    Fg = np.zeros((X.shape[0], S.shape[0]))  # 4x3
    for g in range(S.shape[0]):
        tempSum = np.zeros(X.shape[0])
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * X[:, i]
        Fg[:, g] = tempSum
    # print(Fg)
    Sg = np.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        tempSum = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * np.dot(X[:, i].reshape(
                (X.shape[0], 1)), X[:, i].reshape((1, X.shape[0])))
        Sg[g] = tempSum
    # print(Sg)
    mu = Fg / Zg
    prodmu = np.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        prodmu[g] = np.dot(mu[:, g].reshape((X.shape[0], 1)),
                           mu[:, g].reshape((1, X.shape[0])))
    # print(prodmu)
    # print(np.dot(mu, mu.T).reshape((1, mu.shape[0], mu.shape[0]))) NO, it is wrong
    cov = Sg / Zg.reshape((Zg.size, 1, 1)) - prodmu
    # The following two lines of code are used to model the constraints
    for g in range(S.shape[0]):
        cov[g] = cov[g] * np.eye(cov[g].shape[0])
        U, s, Vh = np.linalg.svd(cov[g])
        s[s < psi] = psi
        cov[g] = np.dot(U, mcol(s)*U.T)
    # print(cov)
    w = Zg/np.sum(Zg)
    # print(w)
    return (w, mu, cov)


def TiedMstep(X, S, posterior):
    psi = 0.01
    # M-step: update the model parameters.
    Zg = np.sum(posterior, axis=1)  # 3
    # print(Zg)
    # Fg = np.array([np.sum(posterior[0, :].reshape(1, posterior.shape[1])* X, axis=1), np.sum(posterior[1, :].reshape(1, posterior.shape[1])* X, axis=1), np.sum(posterior[2, :].reshape(1, posterior.shape[1])*X, axis=1)])
    # print(Fg)
    Fg = np.zeros((X.shape[0], S.shape[0]))  # 4x3
    for g in range(S.shape[0]):
        tempSum = np.zeros(X.shape[0])
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * X[:, i]
        Fg[:, g] = tempSum
    # print(Fg)
    Sg = np.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        tempSum = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[1]):
            tempSum += posterior[g, i] * np.dot(X[:, i].reshape(
                (X.shape[0], 1)), X[:, i].reshape((1, X.shape[0])))
        Sg[g] = tempSum
    # print(Sg)
    mu = Fg / Zg
    prodmu = np.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        prodmu[g] = np.dot(mu[:, g].reshape((X.shape[0], 1)),
                           mu[:, g].reshape((1, X.shape[0])))
    # print(prodmu)
    # print(np.dot(mu, mu.T).reshape((1, mu.shape[0], mu.shape[0]))) NO, it is wrong
    cov = Sg / Zg.reshape((Zg.size, 1, 1)) - prodmu
    # The following two lines of code are used to model the constraints
    tsum = np.zeros((cov.shape[1], cov.shape[2]))
    for g in range(S.shape[0]):
        tsum += Zg[g]*cov[g]
    for g in range(S.shape[0]):
        cov[g] = 1/X.shape[1] * tsum
        U, s, Vh = np.linalg.svd(cov[g])
        s[s < psi] = psi
        cov[g] = np.dot(U, mcol(s)*U.T)
    # print(cov)
    w = Zg/np.sum(Zg)
    # print(w)
    return (w, mu, cov)


def EMalgorithm(X, gmm, solutionFile):
    # The algorithm consists of two steps, E-step and M-step
    # flag is used to exit the while when the difference between the loglikelihoods
    # becomes smaller than 10^(-6)
    flag = True
    # count is used to count iterations
    count = 0
    while(flag):
        count += 1
        # Given the training set and the initial model parameters, compute
        # log marginal densities and sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm)
        # Compute the AVERAGE loglikelihood, by summing all the log densities and
        # dividing by the number of samples (it's as if we're computing a mean)
        loglikelihood1 = np.sum(logdens)/X.shape[1]
        # ------ E-step ----------
        posterior = Estep(logdens, S)
        # ------ M-step ----------
        (w, mu, cov) = Mstep(X, S, posterior)
        for g in range(len(gmm)):
            # Update the model parameters that are in gmm
            gmm[g] = (w[g], mu[:, g].reshape((mu.shape[0], 1)), cov[g])
        # Compute the new log densities and the new sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm)
        loglikelihood2 = np.sum(logdens)/X.shape[1]
        if (loglikelihood2-loglikelihood1 < 10**(-6)):
            flag = False
        if (loglikelihood2-loglikelihood1 < 0):
            print("ERROR, LOG-LIKELIHOOD IS NOT INCREASING")
    # print(count)
    #print("AVG log likelihood:", loglikelihood2)
    if (solutionFile != ''):
        solution = load_gmm(solutionFile)
        print("Solution:", solution)
        print("My estimated parameters:", gmm)    
    return gmm


def DiagEMalgorithm(X, gmm, solutionFile):
    # The algorithm consists of two steps, E-step and M-step
    # flag is used to exit the while when the difference between the loglikelihoods
    # becomes smaller than 10^(-6)
    flag = True
    # count is used to count iterations
    count = 0
    while(flag):
        count += 1
        # Given the training set and the initial model parameters, compute
        # log marginal densities and sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm)
        # Compute the AVERAGE loglikelihood, by summing all the log densities and
        # dividing by the number of samples (it's as if we're computing a mean)
        loglikelihood1 = np.sum(logdens)/X.shape[1]
        # ------ E-step ----------
        posterior = Estep(logdens, S)
        # ------ M-step ----------
        (w, mu, cov) = DiagMstep(X, S, posterior)
        for g in range(len(gmm)):
            # Update the model parameters that are in gmm
            gmm[g] = (w[g], mu[:, g].reshape((mu.shape[0], 1)), cov[g])
        # Compute the new log densities and the new sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm)
        loglikelihood2 = np.sum(logdens)/X.shape[1]
        if (loglikelihood2-loglikelihood1 < 10**(-6)):
            flag = False
        if (loglikelihood2-loglikelihood1 < 0):
            print("ERROR, LOG-LIKELIHOOD IS NOT INCREASING")
    # print(count)
    #print("AVG log likelihood:", loglikelihood2)
    if (solutionFile != ''):
        solution = load_gmm(solutionFile)
        print("Solution:", solution)
        print("My estimated parameters:", gmm)
    return gmm


def TiedEMalgorithm(X, gmm, solutionFile):
    # The algorithm consists of two steps, E-step and M-step
    # flag is used to exit the while when the difference between the loglikelihoods
    # becomes smaller than 10^(-6)
    flag = True
    # count is used to count iterations
    count = 0
    while(flag):
        count += 1
        # Given the training set and the initial model parameters, compute
        # log marginal densities and sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm)
        # Compute the AVERAGE loglikelihood, by summing all the log densities and
        # dividing by the number of samples (it's as if we're computing a mean)
        loglikelihood1 = np.sum(logdens)/X.shape[1]
        # ------ E-step ----------
        posterior = Estep(logdens, S)
        # ------ M-step ----------
        (w, mu, cov) = TiedMstep(X, S, posterior)
        for g in range(len(gmm)):
            # Update the model parameters that are in gmm
            gmm[g] = (w[g], mu[:, g].reshape((mu.shape[0], 1)), cov[g])
        # Compute the new log densities and the new sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm)
        loglikelihood2 = np.sum(logdens)/X.shape[1]
        if (loglikelihood2-loglikelihood1 < 10**(-6)):
            flag = False
        if (loglikelihood2-loglikelihood1 < 0):
            print("ERROR, LOG-LIKELIHOOD IS NOT INCREASING")
    # print(count)
    #print("AVG log likelihood:", loglikelihood2)
    if (solutionFile != ''):
        solution = load_gmm(solutionFile)
        print("Solution:", solution)
        print("My estimated parameters:", gmm)
    return gmm


def GAU_logpdf(x, mu, var):
    # Function that computes the log-density of the dataset and returns it as a
    # 1-dim array
    return (-0.5*np.log(2*np.pi))-0.5*np.log(var)-(((x-mu)**2)/(2*var))


def plotNormalDensityOverNormalizedHistogram(dataset, gmm):
    # Function used to plot the computed normal density over the normalized histogram
    plt.figure()
    plt.hist(dataset, bins=30, edgecolor='black', linewidth=0.5, density=True)
    # Define an array of equidistant 1000 elements between -10 and 5
    XPlot = np.linspace(-10, 5, 1000)
    # We should plot the density, not the log-density, so we need to use np.exp
    y = np.zeros(1000)
    for i in range(len(gmm)):
        y += gmm[i][0]*np.exp(GAU_logpdf(XPlot, gmm[i]
                              [1], gmm[i][2])).flatten()
    plt.plot(XPlot, y,
             color="red", linewidth=3)
    return


def split(GMM):
    alpha = 0.1
    size = len(GMM)
    splittedGMM = []
    for i in range(size):
        U, s, Vh = np.linalg.svd(GMM[i][2])
        # compute displacement vector
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        splittedGMM.append((GMM[i][0]/2, GMM[i][1]+d, GMM[i][2]))
        splittedGMM.append((GMM[i][0]/2, GMM[i][1]-d, GMM[i][2]))
    # print("Splitted GMM", splittedGMM)
    return splittedGMM


def LBGalgorithm(GMM, X, iterations):
    GMM = EMalgorithm(X, GMM, '')
    for i in range(iterations):
        GMM = split(GMM)
        GMM = EMalgorithm(X, GMM, '')
    return GMM


def DiagLBGalgorithm(GMM, X, iterations):
    GMM = DiagEMalgorithm(X, GMM, '')
    for i in range(iterations):
        GMM = split(GMM)
        GMM = DiagEMalgorithm(X, GMM, '')
    return GMM


def TiedLBGalgorithm(GMM, X, iterations):
    GMM = TiedEMalgorithm(X, GMM, '')
    for i in range(iterations):
        GMM = split(GMM)
        GMM = TiedEMalgorithm(X, GMM, '')
    return GMM


def performClassification(DTR0, DTR1, DTR2, DEV, algorithm, K, LEV, constrain):
    # Define a list that includes the three splitted training set
    D = [DTR0, DTR1, DTR2]
    # Define a list to store marginal likelihoods for the three sets
    marginalLikelihoods = []
    # Iterate on the three sets
    for i in range(len(D)):
        wg = 1.0
        # Find mean and covariance matrices, reshape them as matrices because they
        # are scalar and in the following we need them as matrices
        mug = D[i].mean(axis=1).reshape((D[i].shape[0], 1))
        sigmag = constrain(np.cov(D[i]).reshape(
            (D[i].shape[0], D[i].shape[0])))
        # Define initial component
        initialGMM = [(wg, mug, sigmag)]
        finalGMM = algorithm(initialGMM, D[i], K)
        # Compute marginal likelihoods and append them to the list
        marginalLikelihoods.append(logpdf_GMM(DEV, finalGMM)[0])
    # Stack all the likelihoods in PD
    PD = np.vstack(
        (marginalLikelihoods[0], marginalLikelihoods[1], marginalLikelihoods[2]))
    # Compute the predicted labels
    predictedLabels = np.argmax(PD, axis=0)
    numberOfCorrectPredictions = np.array(predictedLabels == LEV).sum()
    accuracy = numberOfCorrectPredictions/LEV.size*100
    errorRate = 100-accuracy
    return errorRate


if __name__ == "__main__":
    # ------------- GAUSSIAN MIXTURE MODELS ------------------
    # Load 4-DIM data
    X4 = np.load("Data/GMM_data_4D.npy")
    # Load reference GMM
    gmm4 = load_gmm("Data/GMM_4D_3G_init.json")
    # Load log-densities results
    logdensres4 = np.load("Data/GMM_4D_3G_init_ll.npy")
    # Compute log-densities
    (logdens4, S4) = logpdf_GMM(X4, gmm4)
    # print(logdens4-logdensres4)
    # Load 1-DIM data
    X1 = np.load("Data/GMM_data_1D.npy")
    # Load reference GMM
    gmm1 = load_gmm("Data/GMM_1D_3G_init.json")
    # Load log-densities results
    logdensres1 = np.load("Data/GMM_1D_3G_init_ll.npy")
    # Compute log-densities
    (logdens1, S1) = logpdf_GMM(X1, gmm1)
    # print(logdens1-logdensres1)
    # # ------------- GMM ESTIMATION: THE EM ALGORITHM --------------
    # # The EM algorithm can be used to estimate the parameters of a GMM that
    # # maximize the likelihood for a training set X.
    # gmm4 = EMalgorithm(X4, gmm4, "Data/GMM_4D_3G_EM.json")
    # gmm1 = EMalgorithm(X1, gmm1, "Data/GMM_1D_3G_EM.json")
    # plotNormalDensityOverNormalizedHistogram(X1.flatten(), gmm1)
    # # ----------- LBG ALGORITHM + CONSTRAINING THE EIGENVALUES OF THE
    # # COVARIANCE MATRICES---------------
    # wg = 1.0
    # # Find mean and covariance matrices, reshape them as matrices because they
    # # are scalar and in the following we need them as matrices
    # mug = X1.mean().reshape((X1.shape[0], X1.shape[0]))
    # sigmag = np.cov(X1).reshape((X1.shape[0], X1.shape[0]))
    # # Define initial component
    # GMM_1 = [(wg, mug, sigmag)]
    # GMM_4 = LBGalgorithm(GMM_1, X1, 2)
    # # Plot
    # plotNormalDensityOverNormalizedHistogram(X1.flatten(), GMM_4)
    # # Load the solution for the 4-component GMM
    # gmm4solution = load_gmm("Data/GMM_1D_4G_EM_LBG.json")
    # # Print the values. The values are very similar, so I guess they're ok (pay attention that
    # # values in GMM_4 and in gmm4solution have different order)
    # # print(GMM_4)
    # # print(gmm4solution)
    # # ----------------- DIAGONAL COVARIANCE GMM-------------------
    # wg = 1.0
    # # Find mean and covariance matrices, reshape them as matrices because they
    # # are scalar and in the following we need them as matrices
    # mug = X1.mean().reshape((X1.shape[0], X1.shape[0]))
    # sigmag = np.cov(X1).reshape((X1.shape[0], X1.shape[0]))
    # # Define initial component
    # GMM_1 = [(wg, mug, sigmag)]
    # GMM_4 = DiagLBGalgorithm(GMM_1, X1, 2)
    # # Plot
    # plotNormalDensityOverNormalizedHistogram(X1.flatten(), GMM_4)
    # -------------- TIED COVARIANCE GMM ---------------
    # wg = 1.0
    # # Find mean and covariance matrices, reshape them as matrices because they
    # # are scalar and in the following we need them as matrices
    # mug = X1.mean().reshape((X1.shape[0], X1.shape[0]))
    # sigmag = np.cov(X1).reshape((X1.shape[0], X1.shape[0]))
    # # Define initial component
    # GMM_1 = [(wg, mug, sigmag)]
    # GMM_4 = TiedLBGalgorithm(GMM_1, X1, 2)
    # # Plot
    # plotNormalDensityOverNormalizedHistogram(X1.flatten(), GMM_4)
    # --------------GMM FOR CLASSIFICATION -----------------
    # Load iris dataset
    D, L = load_iris()
    # D is a 4x150 matrix, L is a 1-dim vector of size 150
    # We can now split the dataset in two parts, the first part will be used
    # for model training, the second for evaluation. For example 100 and 50
    print('Splitting the iris dataset of 150 samples into a training set of 100'
          'samples and an evaluation set of 50 samples...')
    (DTR, LTR), (DEV, LEV) = split_db_2to1(D, L)
    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]
    DTR2 = DTR[:, LTR == 2]
    # Full covariance (standard)
    errorRates = []
    for i in range(5):
        errorRates.append(performClassification(
            DTR0, DTR1, DTR2, DEV, LBGalgorithm, i, LEV, constrainSigma))
    print("Full covariance (standard) error rates:", errorRates)
    # Diagonal covariance
    errorRates = []
    for i in range(5):
        errorRates.append(performClassification(
            DTR0, DTR1, DTR2, DEV, DiagLBGalgorithm, i, LEV, DiagConstrainSigma))
    print("Diagonal covariance error rates:", errorRates)
    # Tied covariance
    errorRates = []
    for i in range(5):
        errorRates.append(performClassification(
            DTR0, DTR1, DTR2, DEV, TiedLBGalgorithm, i, LEV, constrainSigma))
    print("Tied covariance error rates:", errorRates)
