# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 17:16:55 2021

@author: Claudio
"""

import sklearn.datasets as skds
import numpy as np
import scipy as sc


def load_iris():
    # Here is shown another way to load the iris dataset. We can load it from
    # the sklearn library but we need to transpose the data matrix, since we
    # work with a column representation of feature vectors
    D, L = skds.load_iris()['data'].T, skds.load_iris()['target']
    return D, L


def vcol(vector, shape0):
    # Auxiliary function to transform 1-dim vectors to column vectors.
    return vector.reshape(shape0, 1)


def vrow(vector, shape1):
    # Auxiliary function to transform 1-dim vecotrs to row vectors.
    return vector.reshape(1, shape1)


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


def logpdf_GAU_ND(x, mu, sigma):
    # There are two options here, read lab 04 README.md for an explanation
    # return np.diag(-(x.shape[0]/2)*np.log(2*np.pi)-(1/2)*(np.linalg.slogdet(sigma)[1])-(1/2)*np.dot(np.dot((x-mu).T,np.linalg.inv(sigma)),(x-mu)))
    # or
    return -(x.shape[0]/2)*np.log(2*np.pi)-(1/2)*(np.linalg.slogdet(sigma)[1])-(1/2)*((np.dot((x-mu).T, np.linalg.inv(sigma))).T*(x-mu)).sum(axis=0)


def pdf_GAU_ND(x, mu, sigma):
    # There are two options here (for the-log density), read lab 04 README.md
    # for an explanation.  I'm using e^(result using the log density formula).
    return np.exp(np.diag(-(x.shape[0]/2)*np.log(2*np.pi)-(1/2)*(np.linalg.slogdet(sigma)[1])-(1/2)*np.dot(np.dot((x-mu).T, np.linalg.inv(sigma)), (x-mu))))
    # or
    # return np.exp(-(x.shape[0]/2)*np.log(2*np.pi)-(1/2)*(np.linalg.slogdet(sigma)[1])-(1/2)*((np.dot((x-mu).T, np.linalg.inv(sigma))).T*(x-mu)).sum(axis=0))
    # actual non-log formula for the density:
    # return np.diag(np.dot(((2*np.pi)**(-(x.shape[0]/2)))*(np.linalg.det(sigma)**(-1/2)), np.exp((-1/2)*np.dot(np.dot((x-mu).T,np.linalg.inv(sigma)),(x-mu)))))


def computeScoreMatrix(D, mu0, sigma0, mu1, sigma1, mu2, sigma2, callback):
    # Function that computes the score matrix as requested.
    # We're working with a MVG distribution, so we need to compute the density
    # for the MVG. We indeed computed the LOG-density, so maybe we can proceed
    # in two ways:
    # 1) maybe we could do e^(result using the log density formula), I tested it
    # and it works;
    # 2) we could implement the density (we can find it on the slides about)
    # probability, page 62).
    # I implemented all the three methods.
    S = np.array([callback(D, mu0, sigma0), callback(
        D, mu1, sigma1), callback(D, mu2, sigma2)])
    return S


def computeMLestimates(D, L):
    # Compute classes means over columns of the dataset matrix
    mu0 = D[:, L == 0].mean(axis=1)
    mu1 = D[:, L == 1].mean(axis=1)
    mu2 = D[:, L == 2].mean(axis=1)
    # Reshape all of them as 4x1 column vectors
    mu0 = vcol(mu0, mu0.size)
    mu1 = vcol(mu1, mu1.size)
    mu2 = vcol(mu2, mu2.size)
    # Count number of elements in each class
    n0 = D[:, L == 0].shape[1]
    n1 = D[:, L == 1].shape[1]
    n2 = D[:, L == 2].shape[1]
    # Subtract classes means from classes datasets with broadcasting
    DC0 = D[:, L == 0]-mu0
    DC1 = D[:, L == 1]-mu1
    DC2 = D[:, L == 2]-mu2
    # Compute classes covariance matrices
    sigma0 = (1/n0)*(np.dot(DC0, DC0.T))
    sigma1 = (1/n1)*(np.dot(DC1, DC1.T))
    sigma2 = (1/n2)*(np.dot(DC2, DC2.T))
    return (mu0, sigma0), (mu1, sigma1), (mu2, sigma2)


def MVGlogdensities(DTR, LTR, DEV, LEV):
    # As already discussed, it's better to work in the log domain.
    # If we need, we can recover posterior probabilities at the end.
    # Compute estimates for model parameters (empirical mean
    # and covariance matrix of each class). This is the training phase.
    print("Starting training phase on the training set to get maximum "
          "likelihood solution for model parameters...")
    (mu0, sigma0), (mu1, sigma1), (mu2, sigma2) = computeMLestimates(DTR, LTR)
    print("mu0: ", mu0)
    print("mu1: ", mu1)
    print("mu2: ", mu2)
    print("sigma0: ", sigma0)
    print("sigma1: ", sigma1)
    print("sigma2: ", sigma2)
    print("Training phase completed.")
    # Now we have the estimated model parameters and we can turn our attention towards
    # inference for a test sample x of the evaluation set. The final goal is to
    # compute class posterior probabilities, but we split the process in three stages.
    print("Starting evaluation phase on the evaluation set to compute class "
          "posterior probabilities, to predict labels on the evaluation set "
          "and to check accuracy and error rate of the MVG classifier...")
    # 1) Compute, for each test sample, the MVG log-density.
    # We can proceed as seen in lab 04 and we can store class-conditional
    # probabilities (the computed log-densities) in a score matrix logS. logS[i, j]
    # should be the class conditional probability for sample j given class i.
    print("First stage: computing the score matrix logS containing the log-densities...")
    logS = computeScoreMatrix(DEV, mu0, sigma0, mu1,
                              sigma1, mu2, sigma2, logpdf_GAU_ND)
    print("Score matrix logS: ", logS)
    # 2) Compute the matrix of joint log-distribution probabilities logSJoint for
    # samples and classes combining the score matrix with prior information.
    # We assume that the three classes have the same
    # prior probability P(c) = 1/3. logSJoints requires adding each row of
    # logS to the logarithm of the prior probability of the corresponding class.
    print("Second stage: computing the matrix of joint distribution "
          "probabilities SJoint combining the score matrix S with prior "
          "information P(c) = 1/3 for all classes...")
    # Define a new array with the logarithm of prior probabilities and reshape it as a
    # 3x1 column vector.
    priorLogProbabilities = vcol(
        np.array([np.log(1/3), np.log(1/3), np.log(1/3)]), 3)
    # Then add logS to it
    logSJoint = logS+priorLogProbabilities  # 3x50
    # logSJoint is called lc on the pdf
    print("logSJoint matrix of joint log-distribution probabilities: ", logSJoint)
    # 3) We can finally compute class posterior probabilities. But we need to compute
    # the marginal log-density. We can use scipy.special.logsumexp(logSJoint, axis=0)
    print("Third stage: computing class posterior probabilities...")
    marginalLogDensities = vrow(
        sc.special.logsumexp(logSJoint, axis=0), 50)  # 1x50
    # Now we can compute the array of class log-posterior probabilities logSPost.
    logSPost = logSJoint-marginalLogDensities
    # The row vector of marginalLogDensities is subtracted from each row
    # of the logSPost matrix (exploiting broadcasting)
    print("logSPost matrix of class log-posterior probabilities: ", logSPost)
    # Now we have to compute predicted labels according to our class log-posterior
    # probabilities.
    print("Final stage: computing predicted labels and checking classifier "
          "accuracy and error rate...")
    # The predicted label is obtained as the class that has maximum posterior
    # probability, in our 3x50 logSPost matrix. This needs to be done for each sample.
    # We can use argmax with axis=0 on the logSPost matrix. It will return an
    # array whose values are the indices (in our case 0, 1, 2) of the maximum
    # values along the specified axis. (So, for us is the maximum of each column)
    predictedLabels = logSPost.argmax(axis=0)
    print("Predicted labels: ", predictedLabels)
    # We can now compute an array of booleans corresponding to whether predicted
    # and real labels are equal or not. Then, summing all the elements of a
    # boolean array gives the number of elements that are True.
    numberOfCorrectPredictions = np.array(predictedLabels == LEV).sum()
    # Now we can compute percentage values for accuracy and error rate.
    accuracy = numberOfCorrectPredictions/LEV.size*100
    errorRate = 100-accuracy
    print("Accuracy of the MVG classifier when working with log-densities: %.2f %%" % (accuracy))
    print("Error rate: %.2f %%" % (errorRate))
    print("Evaluation phase completed.")
    return


def MVGdensities(DTR, LTR, DEV, LEV):
    # Compute estimates for model parameters (empirical mean
    # and covariance matrix of each class). This is the training phase.
    print("Starting training phase on the training set to get maximum "
          "likelihood solution for model parameters...")
    (mu0, sigma0), (mu1, sigma1), (mu2, sigma2) = computeMLestimates(DTR, LTR)
    print("mu0: ", mu0)
    print("mu1: ", mu1)
    print("mu2: ", mu2)
    print("sigma0: ", sigma0)
    print("sigma1: ", sigma1)
    print("sigma2: ", sigma2)
    print("Training phase completed.")
    # Now we have the estimated model parameters and we can turn our attention towards
    # inference for a test sample x of the evaluation set. The final goal is to
    # compute class posterior probabilities, but we split the process in three stages.
    print("Starting evaluation phase on the evaluation set to compute class "
          "posterior probabilities, to predict labels on the evaluation set "
          "and to check accuracy and error rate of the MVG classifier...")
    # 1) Compute, for each test sample, the MVG density.
    # We can proceed as seen in lab 04 and we can store class-conditional
    # probabilities (the computed densities) in a score matrix S. S[i, j]
    # should be the class conditional probability for sample j given class i.
    print("First stage: computing the score matrix S containing the densities...")
    S = computeScoreMatrix(DEV, mu0, sigma0, mu1,
                           sigma1, mu2, sigma2, pdf_GAU_ND)
    print("Score matrix S: ", S)
    # 2) Compute the matrix of joint distribution probabilities SJoint for
    # samples and classes combining the score matrix with prior information.
    # We assume that the three classes have the same
    # prior probability P(c) = 1/3. SJoints requires multiplying each row of
    # S by the prior probability of the corresponding class.
    print("Second stage: computing the matrix of joint distribution "
          "probabilities SJoint combining the score matrix S with prior "
          "information P(c) = 1/3 for all classes...")
    # Define a new array with the prior probabilities and reshape it as a
    # 3x1 column vector.
    priorProbabilities = vcol(np.array([1/3, 1/3, 1/3]), 3)
    # Then multiply S by it
    SJoint = S*priorProbabilities  # 3x50
    print("SJoint matrix of joint distribution probabilities: ", SJoint)
    # 3) We can finally compute class posterior probabilities. This requires
    # summing over all classes (for each sample) the joint probability, which
    # we have stored in matrix SJoint, to compute the marginal densities.
    print("Third stage: computing class posterior probabilities...")
    marginaldensities = vrow(SJoint.sum(axis=0), 50)  # 1x50
    # Now we can compute the array of class posterior probabilities SPost.
    SPost = SJoint/marginaldensities
    # Each row of SJoint is divided by the row of marginaldensities (exploiting broadcasting)
    print("SPost matrix of class posterior probabilities: ", SPost)
    # Now we have to compute predicted labels according to our class posterior
    # probabilities.
    print("Final stage: computing predicted labels and checking classifier "
          "accuracy and error rate...")
    # The predicted label is obtained as the class that has maximum posterior
    # probability, in our 3x50 SPost matrix. This needs to be done for each sample.
    # We can use argmax with axis=0 on the SPost matrix. It will return an
    # array whose values are the indices (in our case 0, 1, 2) of the maximum
    # values along the specified axis. (So, for us is the maximum of each column)
    predictedLabels = SPost.argmax(axis=0)
    print("Predicted labels: ", predictedLabels)
    # We can now compute an array of booleans corresponding to whether predicted
    # and real labels are equal or not. Then, summing all the elements of a
    # boolean array gives the number of elements that are True.
    numberOfCorrectPredictions = np.array(predictedLabels == LEV).sum()
    # Now we can compute percentage values for accuracy and error rate.
    accuracy = numberOfCorrectPredictions/LEV.size*100
    errorRate = 100-accuracy
    print("Accuracy of the MVG classifier when working with densities: %.2f %%" % (accuracy))
    print("Error rate: %.2f %%" % (errorRate))
    print("Evaluation phase completed.")
    return


def multivariateGaussianClassifier(DTR, LTR, DEV, LEV):
    print("WORKING ON MVG DENSITIES")
    MVGdensities(DTR, LTR, DEV, LEV)
    print("MVG DENSITIES - END")
    print("WORKING ON MVG LOG-DENSITIES")
    MVGlogdensities(DTR, LTR, DEV, LEV)
    print("MVG LOG-DENSITIES - END")
    return


def kfoldCrossValidationMVG(D, L):
    # Same procedure of MVGlogdensities but with leave-one-out approach:
    # number of folds is equal to the number of samples, we iterate over all samples
    # and each time we use one of them as evaluation set and the remaining 149
    # as training set.
    # Then we compute the predicted lable and count how many times it is correct.
    numberOfCorrectPredictions = 0
    for i in range(D.shape[1]):  # i=0...149
        # Each time we need to delete the i-th column (sample) from the training set
        # while in the evaluation set we should keep only the i-th column.
        # vcol is necessary to exploit broadcasting
        (DTR, LTR), (DEV, LEV) = (np.delete(D, i, 1),
                                  np.delete(L, i)), (vcol(D[:, i], 4), L[i])
        (mu0, sigma0), (mu1, sigma1), (mu2, sigma2) = computeMLestimates(DTR, LTR)
        logS = computeScoreMatrix(DEV, mu0, sigma0, mu1,
                                  sigma1, mu2, sigma2, logpdf_GAU_ND)
        priorLogProbabilities = vcol(
            np.array([np.log(1/3), np.log(1/3), np.log(1/3)]), 3)
        logSJoint = logS+priorLogProbabilities  # 3x1
        marginalLogDensities = vrow(
            sc.special.logsumexp(logSJoint, axis=0), 1)  # 1x1
        # We need to keep it as 1x1 to exploit broadcasting
        logSPost = logSJoint-marginalLogDensities
        predictedLabels = logSPost.argmax(axis=0)
        # Add the result of prediction to the proper variable
        numberOfCorrectPredictions += np.array(predictedLabels == LEV).sum()
    # Now we can compute percentage values for accuracy and error rate.
    accuracy = numberOfCorrectPredictions/L.size * \
        100  # over all samples, that are 150
    errorRate = 100-accuracy
    print("Accuracy of the MVG classifier when working with log-densities and "
          "using K-fold cross validation with leave-one-out approach: %.2f %%" % (accuracy))
    print("Error rate: %.2f %%" % (errorRate))
    return


def kfoldCrossValidationNBG(D, L):
    # Same procedure of naiveBayesGaussianClassifier but with leave-one-out approach:
    # number of folds is equal to the number of samples, we iterate over all samples
    # and each time we use one of them as evaluation set and the remaining 149
    # as training set.
    # Then we compute the predicted lable and count how many times it is correct.
    numberOfCorrectPredictions = 0
    for i in range(D.shape[1]):  # i=0...149
        # Each time we need to delete the i-th column (sample) from the training set
        # while in the evaluation set we should keep only the i-th column.
        # vcol is necessary to exploit broadcasting
        (DTR, LTR), (DEV, LEV) = (np.delete(D, i, 1),
                                  np.delete(L, i)), (vcol(D[:, i], 4), L[i])
        (mu0, sigma0), (mu1, sigma1), (mu2, sigma2) = computeMLestimates(DTR, LTR)
        (sigma0, sigma1, sigma2) = (sigma0*np.identity(sigma0.shape[0]), sigma1*np.identity(
            sigma1.shape[0]), sigma2*np.identity(sigma2.shape[0]))
        logS = computeScoreMatrix(DEV, mu0, sigma0, mu1,
                                  sigma1, mu2, sigma2, logpdf_GAU_ND)
        priorLogProbabilities = vcol(
            np.array([np.log(1/3), np.log(1/3), np.log(1/3)]), 3)
        logSJoint = logS+priorLogProbabilities  # 3x1
        marginalLogDensities = vrow(
            sc.special.logsumexp(logSJoint, axis=0), 1)  # 1x1
        # We need to keep it as 1x1 to exploit broadcasting
        logSPost = logSJoint-marginalLogDensities
        predictedLabels = logSPost.argmax(axis=0)
        # Add the result of prediction to the proper variable
        numberOfCorrectPredictions += np.array(predictedLabels == LEV).sum()
    # Now we can compute percentage values for accuracy and error rate.
    accuracy = numberOfCorrectPredictions/L.size * \
        100  # over all samples, that are 150
    errorRate = 100-accuracy
    print("Accuracy of the Naive Bayes Gaussian classifier when working with log-densities and "
          "using K-fold cross validation with leave-one-out approach: %.2f %%" % (accuracy))
    print("Error rate: %.2f %%" % (errorRate))
    return


def kfoldCrossValidationTCG(D, L):
    # Same procedure of tiedCovarianceGaussianClassifier but with leave-one-out approach:
    # number of folds is equal to the number of samples, we iterate over all samples
    # and each time we use one of them as evaluation set and the remaining 149
    # as training set.
    # Then we compute the predicted lable and count how many times it is correct.
    numberOfCorrectPredictions = 0
    for i in range(D.shape[1]):  # i=0...149
        # Each time we need to delete the i-th column (sample) from the training set
        # while in the evaluation set we should keep only the i-th column.
        # vcol is necessary to exploit broadcasting
        (DTR, LTR), (DEV, LEV) = (np.delete(D, i, 1),
                                  np.delete(L, i)), (vcol(D[:, i], 4), L[i])
        (mu0, sigma0), (mu1, sigma1), (mu2, sigma2) = computeMLestimates(DTR, LTR)
        sigma = (1/DTR.shape[1])*((LTR == 0).sum()*sigma0 +
                                  (LTR == 1).sum()*sigma1+(LTR == 2).sum()*sigma2)
        logS = computeScoreMatrix(DEV, mu0, sigma, mu1,
                                  sigma, mu2, sigma, logpdf_GAU_ND)
        priorLogProbabilities = vcol(
            np.array([np.log(1/3), np.log(1/3), np.log(1/3)]), 3)
        logSJoint = logS+priorLogProbabilities  # 3x1
        marginalLogDensities = vrow(
            sc.special.logsumexp(logSJoint, axis=0), 1)  # 1x1
        # We need to keep it as 1x1 to exploit broadcasting
        logSPost = logSJoint-marginalLogDensities
        predictedLabels = logSPost.argmax(axis=0)
        # Add the result of prediction to the proper variable
        numberOfCorrectPredictions += np.array(predictedLabels == LEV).sum()
    # Now we can compute percentage values for accuracy and error rate.
    accuracy = numberOfCorrectPredictions/L.size * \
        100  # over all samples, that are 150
    errorRate = 100-accuracy
    print("Accuracy of the Tied Covariance Gaussian classifier when working with log-densities and "
          "using K-fold cross validation with leave-one-out approach: %.2f %%" % (accuracy))
    print("Error rate: %.2f %%" % (errorRate))
    return


def naiveBayesGaussianClassifier(DTR, LTR, DEV, LEV):
    print("WORKING ON NAIVE BAYES GAUSSIAN CLASSIFIER LOG-DENSITIES")
    # Compute estimates for model parameters (empirical mean
    # and covariance matrix of each class). This is the training phase.
    print("Starting training phase on the training set to get maximum "
          "likelihood solution for model parameters...")
    (mu0, sigma0), (mu1, sigma1), (mu2, sigma2) = computeMLestimates(DTR, LTR)
    # But for the Naive Bayes version of MVG the covariance matrices are diagonal!
    # The ML solution for the mean parameters is the same, while the ML solution
    # for the covariance matrices is different. Since the number of feature is
    # small, we can adapt the MVG code by simply zeroing the out-of-diagonal
    # elements of the MVG ML solution, keeping them as matrices. This can be
    # done, for example, multiplying element-wise the MVG ML solution with the
    # identity matrix. The rest of the code is the same as MVGlogdensities.
    # This procedure (keeping the matrix, etc.) is not advisable when we work
    # on large dimensional data, in those cases it may be better to implement
    # ad-hoc functions to work directly with just the diagonal of the covariance
    # matrices.
    (sigma0, sigma1, sigma2) = (sigma0*np.identity(sigma0.shape[0]), sigma1*np.identity(
        sigma1.shape[0]), sigma2*np.identity(sigma2.shape[0]))
    print("mu0: ", mu0)
    print("mu1: ", mu1)
    print("mu2: ", mu2)
    print("sigma0 with zeros for elements out-of-diagonal: ", sigma0)
    print("sigma1 with zeros for elements out-of-diagonal: ", sigma1)
    print("sigma2 with zeros for elements out-of-diagonal: ", sigma2)
    print("Training phase completed.")
    # Now we have the estimated model parameters and we can turn our attention towards
    # inference for a test sample x of the evaluation set. The final goal is to
    # compute class posterior probabilities, but we split the process in three stages.
    print("Starting evaluation phase on the evaluation set to compute class "
          "posterior probabilities, to predict labels on the evaluation set "
          "and to check accuracy and error rate of the Naive Bayes Gaussian classifier...")
    # 1) Compute, for each test sample, the MVG log-density.
    # We can proceed as seen in lab 04 and we can store class-conditional
    # probabilities (the computed log-densities) in a score matrix logS. logS[i, j]
    # should be the class conditional probability for sample j given class i.
    print("First stage: computing the score matrix logS containing the log-densities...")
    logS = computeScoreMatrix(DEV, mu0, sigma0, mu1,
                              sigma1, mu2, sigma2, logpdf_GAU_ND)
    print("Score matrix logS: ", logS)
    # 2) Compute the matrix of joint log-distribution probabilities logSJoint for
    # samples and classes combining the score matrix with prior information.
    # We assume that the three classes have the same
    # prior probability P(c) = 1/3. logSJoints requires adding each row of
    # logS to the logarithm of the prior probability of the corresponding class.
    print("Second stage: computing the matrix of joint distribution "
          "probabilities SJoint combining the score matrix S with prior "
          "information P(c) = 1/3 for all classes...")
    # Define a new array with the logarithm of prior probabilities and reshape it as a
    # 3x1 column vector.
    priorLogProbabilities = vcol(
        np.array([np.log(1/3), np.log(1/3), np.log(1/3)]), 3)
    # Then add logS to it
    logSJoint = logS+priorLogProbabilities  # 3x50
    # logSJoint is called lc on the pdf
    print("logSJoint matrix of joint log-distribution probabilities: ", logSJoint)
    # 3) We can finally compute class posterior probabilities. But we need to compute
    # the marginal log-density. We can use scipy.special.logsumexp(logSJoint, axis=0)
    print("Third stage: computing class posterior probabilities...")
    marginalLogDensities = vrow(
        sc.special.logsumexp(logSJoint, axis=0), 50)  # 1x50
    # Now we can compute the array of class log-posterior probabilities logSPost.
    logSPost = logSJoint-marginalLogDensities
    # The row vector of marginalLogDensities is subtracted from each row
    # of the logSPost matrix (exploiting broadcasting)
    print("logSPost matrix of class log-posterior probabilities: ", logSPost)
    # Now we have to compute predicted labels according to our class log-posterior
    # probabilities.
    print("Final stage: computing predicted labels and checking classifier "
          "accuracy and error rate...")
    # The predicted label is obtained as the class that has maximum posterior
    # probability, in our 3x50 logSPost matrix. This needs to be done for each sample.
    # We can use argmax with axis=0 on the logSPost matrix. It will return an
    # array whose values are the indices (in our case 0, 1, 2) of the maximum
    # values along the specified axis. (So, for us is the maximum of each column)
    predictedLabels = logSPost.argmax(axis=0)
    print("Predicted labels: ", predictedLabels)
    # We can now compute an array of booleans corresponding to whether predicted
    # and real labels are equal or not. Then, summing all the elements of a
    # boolean array gives the number of elements that are True.
    numberOfCorrectPredictions = np.array(predictedLabels == LEV).sum()
    # Now we can compute percentage values for accuracy and error rate.
    accuracy = numberOfCorrectPredictions/LEV.size*100
    errorRate = 100-accuracy
    print("Accuracy of the Naive Bayes Gaussian classifier when working with log-densities: %.2f %%" % (accuracy))
    print("Error rate: %.2f %%" % (errorRate))
    print("Evaluation phase completed.")
    print("NAIVE BAYES GAUSSIAN CLASSIFIER LOG-DENSITIES - END")
    return


def tiedCovarianceGaussianClassifier(DTR, LTR, DEV, LEV):
    print("WORKING ON TIED COVARIANCE GAUSSIAN CLASSIFIER LOG-DENSITIES")
    # Compute estimates for model parameters (empirical mean
    # and covariance matrix of each class). This is the training phase.
    print("Starting training phase on the training set to get maximum "
          "likelihood solution for model parameters...")
    (mu0, sigma0), (mu1, sigma1), (mu2, sigma2) = computeMLestimates(DTR, LTR)
    # But for the tied covariance version of the classifier the class covariance
    # matrices are tied, this mean that sigmai=sigma, they're all the same.
    # We have seen that the ML solution for the class means is again the same.
    # The ML solution for the covariance matrix, instead, is given by the empirical
    # within-class covariance matrix. We already computed it when we implemented
    # LDA, alternatively (and I will do so in the following) we can compute it
    # from the covariance matrices sigma0, sigma1 and sigma2:
    sigma = (1/DTR.shape[1])*((LTR == 0).sum()*sigma0 +
                              (LTR == 1).sum()*sigma1+(LTR == 2).sum()*sigma2)
    print("mu0: ", mu0)
    print("mu1: ", mu1)
    print("mu2: ", mu2)
    print("sigma0: ", sigma0)
    print("sigma1: ", sigma1)
    print("sigma2: ", sigma2)
    print("sigma, within-class covariance matrix: ", sigma)
    print("Training phase completed.")
    # Now we have the estimated model parameters and we can turn our attention towards
    # inference for a test sample x of the evaluation set. The final goal is to
    # compute class posterior probabilities, but we split the process in three stages.
    print("Starting evaluation phase on the evaluation set to compute class "
          "posterior probabilities, to predict labels on the evaluation set "
          "and to check accuracy and error rate of the Tied Covariance Gaussian classifier...")
    # 1) Compute, for each test sample, the MVG log-density.
    # We can proceed as seen in lab 04 and we can store class-conditional
    # probabilities (the computed log-densities) in a score matrix logS. logS[i, j]
    # should be the class conditional probability for sample j given class i.
    print("First stage: computing the score matrix logS containing the log-densities...")
    logS = computeScoreMatrix(DEV, mu0, sigma, mu1,
                              sigma, mu2, sigma, logpdf_GAU_ND)
    print("Score matrix logS: ", logS)
    # 2) Compute the matrix of joint log-distribution probabilities logSJoint for
    # samples and classes combining the score matrix with prior information.
    # We assume that the three classes have the same
    # prior probability P(c) = 1/3. logSJoints requires adding each row of
    # logS to the logarithm of the prior probability of the corresponding class.
    print("Second stage: computing the matrix of joint distribution "
          "probabilities SJoint combining the score matrix S with prior "
          "information P(c) = 1/3 for all classes...")
    # Define a new array with the logarithm of prior probabilities and reshape it as a
    # 3x1 column vector.
    priorLogProbabilities = vcol(
        np.array([np.log(1/3), np.log(1/3), np.log(1/3)]), 3)
    # Then add logS to it
    logSJoint = logS+priorLogProbabilities  # 3x50
    # logSJoint is called lc on the pdf
    print("logSJoint matrix of joint log-distribution probabilities: ", logSJoint)
    # 3) We can finally compute class posterior probabilities. But we need to compute
    # the marginal log-density. We can use scipy.special.logsumexp(logSJoint, axis=0)
    print("Third stage: computing class posterior probabilities...")
    marginalLogDensities = vrow(
        sc.special.logsumexp(logSJoint, axis=0), 50)  # 1x50
    # Now we can compute the array of class log-posterior probabilities logSPost.
    logSPost = logSJoint-marginalLogDensities
    # The row vector of marginalLogDensities is subtracted from each row
    # of the logSPost matrix (exploiting broadcasting)
    print("logSPost matrix of class log-posterior probabilities: ", logSPost)
    # Now we have to compute predicted labels according to our class log-posterior
    # probabilities.
    print("Final stage: computing predicted labels and checking classifier "
          "accuracy and error rate...")
    # The predicted label is obtained as the class that has maximum posterior
    # probability, in our 3x50 logSPost matrix. This needs to be done for each sample.
    # We can use argmax with axis=0 on the logSPost matrix. It will return an
    # array whose values are the indices (in our case 0, 1, 2) of the maximum
    # values along the specified axis. (So, for us is the maximum of each column)
    predictedLabels = logSPost.argmax(axis=0)
    print("Predicted labels: ", predictedLabels)
    # We can now compute an array of booleans corresponding to whether predicted
    # and real labels are equal or not. Then, summing all the elements of a
    # boolean array gives the number of elements that are True.
    numberOfCorrectPredictions = np.array(predictedLabels == LEV).sum()
    # Now we can compute percentage values for accuracy and error rate.
    accuracy = numberOfCorrectPredictions/LEV.size*100
    errorRate = 100-accuracy
    print("Accuracy of the MVG classifier when working with log-densities: %.2f %%" % (accuracy))
    print("Error rate: %.2f %%" % (errorRate))
    print("Evaluation phase completed.")
    print("NAIVE BAYES GAUSSIAN CLASSIFIER LOG-DENSITIES - END")
    return


if __name__ == "__main__":
    # Load iris dataset
    D, L = load_iris()
    # D is a 4x150 matrix, L is a 1-dim vector of size 150
    # We can now split the dataset in two parts, the first part will be used
    # for model training, the second for evaluation. For example 100 and 50
    print('Splitting the iris dataset of 150 samples into a training set of 100'
          'samples and an evaluation set of 50 samples...')
    (DTR, LTR), (DEV, LEV) = split_db_2to1(D, L)
    # 1st model: MVG classifier with MVG distribution (density domain,
    # log-density domain)
    multivariateGaussianClassifier(DTR, LTR, DEV, LEV)
    # 2nd model: Naive Bayes Gaussian classifier (log-density domain)
    naiveBayesGaussianClassifier(DTR, LTR, DEV, LEV)
    # 3rd model: Tied Covariance Gaussian Classifier (log-density domain)
    tiedCovarianceGaussianClassifier(DTR, LTR, DEV, LEV)
    # K-fold cross validation with leave-one-out approach on MVG distribution
    # (log-density domain)
    kfoldCrossValidationMVG(D, L)
    # K-fold cross validation with leave-one-out approach on Naive Bayes
    # Gaussian classifier (log-density domain)
    kfoldCrossValidationNBG(D, L)
    # K-fold cross validation with leave-one-out approach on Tied Covariance
    # Gaussian classifier (log-density domain)
    kfoldCrossValidationTCG(D, L)
