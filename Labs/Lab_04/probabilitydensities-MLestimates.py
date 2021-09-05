# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 20:45:45 2021

@author: Claudio
"""

import numpy as np
import matplotlib.pyplot as plt


def plotNormalizedHistogramOfDataset(dataset):
    # Function used to plot a normalized histogram for the given dataset
    plt.figure()
    plt.hist(dataset, bins=50, density=True)
    plt.show()
    return


def plotNormalDensityOverNormalizedHistogram(dataset, mu, var):
    # Function used to plot the computed normal density over the normalized histogram
    plt.figure()
    plt.hist(dataset, bins=50, density=True)
    # Define an array of equidistant 1000 elements between -8 and 12
    XPlot = np.linspace(-8, 12, 1000)
    # We should plot the density, not the log-density, so we need to use np.exp
    plt.plot(XPlot, np.exp(GAU_logpdf(XPlot, mu, var)),
             color="red", linewidth=3)
    return


def GAU_pdf(x, mu, var):
    # Function that computes the normal density of the dataset and returns it
    # as a 1-dim array
    return (1/np.sqrt(2*np.pi*var))*np.exp(-(((x-mu)**2)/(2*var)))


def GAU_logpdf(x, mu, var):
    # Function that computes the log-density of the dataset and returns it as a
    # 1-dim array
    return (-0.5*np.log(2*np.pi))-0.5*np.log(var)-(((x-mu)**2)/(2*var))


def computeLikelihood(dataset, mu, var):
    # Function that computes the likelihood for a dataset.
    # Returns the computed likelihood
    ll_samples = GAU_pdf(dataset, mu, var)
    return ll_samples.prod()


def computeLogLikelihood(dataset, mu, var):
    # Function that computes the log-likelihood for a dataset.
    # Returns the computed log-likelihood
    lll_samples = GAU_logpdf(dataset, mu, var)
    return lll_samples.sum()  # This time it's a sum, not a product


def computeMaximumLikelihoodEstimates(dataset):
    return dataset.mean(), dataset.var()


def GaussianDensityEstimation(dataset):
    # Uncomment the following line to plot the normalized histogram of the dataset
    # I commented it because we will see the complete plot at the end
    # plotNormalizedHistogramOfDataset(dataset)
    # Try to get the likelihood. Mu and var are now arbitrary values
    likelihood = computeLikelihood(dataset, 1.0, 2.0)
    print(likelihood)
    # We get 0 from this print. Why? Because values have been normalized,
    # they're all < 1, so by doing their product inside the function we get
    # something very small that at the end saturates to the zero value.
    # To avoid this situation we can work with logarithms. Try to get the log-likelihood
    logLikelihood = computeLogLikelihood(dataset, 1.0, 2.0)
    print(logLikelihood)
    # The log-likelihood does not saturate, it's a negative value but it's ok.
    # We can use this value to compare likelihoods for different model parameters.
    muML, varML = computeMaximumLikelihoodEstimates(dataset)
    logLikelihood = computeLogLikelihood(dataset, muML, varML)
    print(logLikelihood)
    # Now plot the density on top of the histogram.
    plotNormalDensityOverNormalizedHistogram(dataset, muML, varML)
    return


def logpdf_GAU_ND(x, mu, sigma):
    # There are two options here, read README.md for an explanation
    # return np.diag(-(x.shape[0]/2)*np.log(2*np.pi)-(1/2)*(np.linalg.slogdet(sigma)[1])-(1/2)*np.dot(np.dot((x-mu).T,np.linalg.inv(sigma)),(x-mu)))
    # or
    return -(x.shape[0]/2)*np.log(2*np.pi)-(1/2)*(np.linalg.slogdet(sigma)[1])-(1/2)*((np.dot((x-mu).T, np.linalg.inv(sigma))).T*(x-mu)).sum(axis=0)


def MultivariateGaussian():
    # Function to check if logpdf_GAU_ND has been correctly implemented
    # Load new dataset. It's a 2x10 matrix
    XND = np.load('Solution/XND.npy')
    # Load mu vector. It's a 2x1 column vector
    mu = np.load('Solution/muND.npy')
    # Load covariance matrix. It's a 2x2 matrix
    C = np.load('Solution/CND.npy')
    # Load solution of multivariate gaussian (MVG) density for the given data
    pdfSol = np.load('Solution/llND.npy')
    # Compute MVG density using my implementation
    pdfGau = logpdf_GAU_ND(XND, mu, C)
    # In both cases we will get a 1-D array (10,)
    # We compute the distance between the two arrays and the result will be 0 =>
    # =>Everything works fine
    print(np.abs(pdfSol-pdfGau).mean())
    return


if __name__ == "__main__":
    # Load data from file
    XGAU = np.load("Data/XGAU.npy")
    # XGAU is a 1-D array of shape (10000,). Since for this part we work with
    # 1-dim data, we use 1-D arrays rather than row vectors
    GaussianDensityEstimation(XGAU)
    # In this case we have a good fit of the histogram and the density well
    # represents the distribution of our data. If we reduce the number of samples
    # used to estimate the ML parameters the fit won't be as good as it is now.
    # We can uncomment the following lines to try, respectively, with datasets
    # of 9000 elements, 5000 elements, 1000 elements and 100 elements.
    # XGAU1=np.delete(XGAU,range(1000),0)
    # GaussianDensityEstimation(XGAU1)
    # XGAU1=np.delete(XGAU,range(5000),0)
    # GaussianDensityEstimation(XGAU1)
    # XGAU1=np.delete(XGAU,range(9000),0)
    # GaussianDensityEstimation(XGAU1)
    # XGAU1=np.delete(XGAU,range(9900),0)
    # GaussianDensityEstimation(XGAU1)
    MultivariateGaussian()
