# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 13:42:16 2021

@author: Claudio
"""

import scipy.optimize as scopt
import numpy as np
import sklearn.datasets as skds


def f(x):
    # Implementation of the f(y,z) function
    y = x[0]
    z = x[1]
    return (y+3)**2+np.sin(y)+(z+1)**2


def fg(x):
    # Function that returns f(y,z) and its gradient
    y = x[0]
    z = x[1]
    f = (y+3)**2+np.sin(y)+(z+1)**2
    gfy = 2*(y+3)+np.cos(y)
    gfz = 2*(z+1)
    return (f, np.array([gfy, gfz]))


def J(w, b, DTR, LTR, lambd):
    # The computation of log(1+x) can lead to numerical issues when x is small,
    # since the sum will make the contribution of x disappear. We can avoid the
    # issue using function np.log1p which computes log(1+x) in a numerically more
    # stable way. ATTENTION, the term 1+ is already included, we just need to pass
    # x as argument.
    normTerm = lambd/2*(np.linalg.norm(w)**2)
    sumTerm = 0
    for i in range(DTR.shape[1]):
        sumTerm += LTR[i] * np.log1p(np.exp(-np.dot(w.T, DTR[:, i])-b)) + \
            (1-LTR[i])*np.log1p(np.exp(np.dot(w.T, DTR[:, i])+b))
    return normTerm + (1/DTR.shape[1])*sumTerm


def load_iris_binary():
    D, L = skds.load_iris()['data'].T, skds.load_iris()[
        'target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2)
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


def logreg_obj(v, DTR, LTR, l):
    # This function should receive a single numpy array v with shape (D+1),
    # where D is the dimensionality of the feature space (example, D=4 for IRIS,
    # 4 features). v should pack all model parameters (w and b), then we can unpack
    # the array here.
    w, b = v[0:-1], v[-1]
    # The function has to access also DTR, LTR and lambda, which are required
    # to compute the objective. We can choose different strategies, but I
    # choose the easiest way.
    j = J(w, b, DTR, LTR, l)
    return j


if __name__ == "__main__":
    # --------------- NUMERICAL OPTIMIZATION ----------------
    # We're going to use the L-BFGS algorithm. It builds an incremental
    # approximation of the Hessian that is used to identify a search direction
    # pt at each iteration. The algorithm then proceeds at finding an acceptable
    # step size alpha_t for the search direction pt, and uses the direction and
    # step size to update the solution. The algorithm is implemented in scipy
    # (requires importing scipy.optimize) and we'll use the scipy.optimize.fmin_l_bfgs_b
    # interface to the numerical solver.
    # This interface requires at least 2 arguments, func (the function that we want to
    # minimize), x0 (the starting value for the algorithm).
    # This algorithm requires computing the objective function and its gradient.
    # To pass the gradient we have different options:
    # 1) through func, if func returns a tuple with the value and the gradient,
    #    for example ( f(x), grad_x(f(x)) )
    # 2) Through the optional parameter fprime, which is a function computing the
    #    gradient. In this case, func should only return the objective value f(x)
    # 3) Let the implementation compute an approximated gradient by passing approx_grad=True
    #    Also in this case func should only return the objective value f(x)
    # The last option seems the best if we're lazy but it has some problems:
    # 1) The gradient computed through finite differences may not be accurate enough
    # 2) The computations are much more expensive, since we need to evaluate the objective
    #    function a number of times at least D, where D is the size of x, at each iteration,
    #    and if we want a more accurate approximation of the gradient we may need to evaluate
    #    f many more times.
    # The optional argument iprint=1 allows visualizing the iterations of the algorithm.
    (x, f, d) = scopt.fmin_l_bfgs_b(
        f, np.array([0, 0]), approx_grad=True, iprint=1)
    # The function returns a tuple where x is the estimated position of the minimum, f is the
    # objective value at the minimum, d contains additional information
    print("----AUTO GRADIENT, APPROXIMATED----")
    print("Estimated position of the minimum:")
    print(x)
    print("Objective value at the minimum:")
    print(f)
    print("Additional information:")
    print(d)
    print("----GRADIENT BY HAND----")
    (x, f, d) = scopt.fmin_l_bfgs_b(fg, np.array([0, 0]), iprint=1)
    print("Estimated position of the minimum:")
    print(x)
    print("Objective value at the minimum:")
    print(f)
    print("Additional information:")
    print(d)
    # In this case the numerical approximation is good enough. However, if we check the
    # values of the third returned value d, we can see that 'funcalls' (which provides
    # the number of times f was called) is higher in the first case. The numerical
    # approximation of the gradient is significantly more expensive, and the cost becomes
    # worse when the dimensionality of the domain of f increases.
    # --------------BINARY LOGISTIC REGRESSION--------------
    # We'll discriminate between iris virginica and iris versicolor. We will
    # ignore iris setosa. We will represent labels with 1 (iris versicolor) and
    # 0 (iris virginica).
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    # We implement logistic regression using expression (3). We need to write
    # a function logreg_obj that, given w and b, allows computing J(w, b). Then
    # we can provide this function to the numerical solver to obtain the minimizer
    # of J.
    # Lambda is a hyper-parameter. As usual we should employ  a validation set to
    #estimate good values, but for this laboratory we can simply try different
    #values and see how this affects the performance.
    # 1) lambda = 0 (not stable results!)
    (x, f, d) = scopt.fmin_l_bfgs_b(logreg_obj,
                                    np.zeros(DTR.shape[0] + 1), args=(DTR, LTR, 0), approx_grad=True)
    # x will contain the estimated values for w and b
    # Once we have trained the model we can compute posterior log-likelihood ratios
    # by simply computing, for each test sample, the score. So we have the array
    # of scores S:
    S = np.dot(x[0:4], DTE) + x[4]
    # Then we can compute class assignments by thresholding the scores with 0
    # (S[i]>0 => LP[i]=1 where LP is the array of predicted labels for the test
    # samples)
    LP = S>0
    # We can now compute an array of booleans corresponding to whether predicted
    # and real labels are equal or not. Then, summing all the elements of a
    # boolean array gives the number of elements that are True.
    numberOfCorrectPredictions = np.array(LP == LTE).sum()
    # Now we can compute percentage values for accuracy and error rate.
    accuracy = numberOfCorrectPredictions/LTE.size*100
    errorRate = 100-accuracy
    print("Lambda=0, value of the objective function= %.5f, error rate=%.1f %%" % (f, errorRate))
    # 2) lambda = 10^-6
    (x, f, d) = scopt.fmin_l_bfgs_b(logreg_obj, np.zeros(
        DTR.shape[0] + 1), args=(DTR, LTR, 0.000001), approx_grad=True)
    S = np.dot(x[0:4], DTE) + x[4]
    LP = S>0
    numberOfCorrectPredictions = np.array(LP == LTE).sum()
    accuracy = numberOfCorrectPredictions/LTE.size*100
    errorRate = 100-accuracy
    print("Lambda=0.000001, value of the objective function= %.5f, error rate=%.1f %%" % (f, errorRate))
    # 3) lambda = 10^-3
    (x, f, d) = scopt.fmin_l_bfgs_b(logreg_obj, np.zeros(
        DTR.shape[0] + 1), args=(DTR, LTR, 0.001), approx_grad=True)
    S = np.dot(x[0:4], DTE) + x[4]
    LP = S>0
    numberOfCorrectPredictions = np.array(LP == LTE).sum()
    accuracy = numberOfCorrectPredictions/LTE.size*100
    errorRate = 100-accuracy
    print("Lambda=0.001, value of the objective function= %.5f, error rate=%.1f %%" % (f, errorRate))
    # 4) lambda = 1
    (x, f, d) = scopt.fmin_l_bfgs_b(logreg_obj, np.zeros(
        DTR.shape[0] + 1), args=(DTR, LTR, 1.0), approx_grad=True)
    S = np.dot(x[0:4], DTE) + x[4]
    LP = S>0
    numberOfCorrectPredictions = np.array(LP == LTE).sum()
    accuracy = numberOfCorrectPredictions/LTE.size*100
    errorRate = 100-accuracy
    print("Lambda=1.0, value of the objective function= %.5f, error rate=%.1f %%" % (f, errorRate))
    