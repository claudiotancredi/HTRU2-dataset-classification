# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 22:09:23 2021

@author: Claudio
"""

import numpy as np
import scipy.linalg as sc
import matplotlib.pyplot as plt


def load(filename):
    list_of_columns = []
    list_of_labels = []
    labels_mapping = {"Iris-setosa": 0,
                      "Iris-versicolor": 1, "Iris-virginica": 2}
    with open(filename, 'r') as f:
        for line in f:
            data = line.split(',')
            if len(data) == 5:
                # This check is necessary to avoid the last line where there is only a \n
                for i in range(len(data)-1):
                    data[i] = float(data[i])
                    # Convert values to float
                # Delete \n at the end of the line
                data[4] = data[4].rstrip('\n')
                # Now create a 1-dim array and reshape it as a column vector,
                # then append it to the appropriate list
                list_of_columns.append(np.array(data[0:4]).reshape((4, 1)))
                # Append the value of the class to the appropriate list
                list_of_labels.append(labels_mapping[data[4]])
    # We have column vectors, we need to create a 4x150 matrix, so we have to
    # stack horizontally all the column vectors
    dataset_matrix = np.hstack(list_of_columns[:])
    # Create a 1-dim array with class values
    class_label_array = np.array(list_of_labels)
    return dataset_matrix, class_label_array


def vcol(vector, shape0):
    # Auxiliary function to transform 1-dim vectors to column vectors.
    return vector.reshape(shape0, 1)


def vrow(vector, shape1):
    # Auxiliary function to transform 1-dim vecotrs to row vectors.
    return vector.reshape(1, shape1)


def custom_scatter(i, j, D, L):
    # Function used for scatter plots. It receives the indexes i, j of the
    # principal components/directions to plot, the projection matrix D and the
    # array L with the values for the classes
    plt.scatter(D[i, L == 0], D[j, L == 0], color="#1e90ff")
    plt.scatter(D[i, L == 1], D[j, L == 1], color="#ff8c00")
    plt.scatter(D[i, L == 2], D[j, L == 2], color="#90ee90")
    plt.legend(["Setosa", "Versicolor", "Virginica"])
    plt.show()
    return


def computePCA(C, D, L):
    # Get the eigenvalues (s) and eigenvectors (columns of U) of C
    s, U = np.linalg.eigh(C)
    # We can start from m=3
    m = D.shape[0]-1
    while(m > 0):
        # Principal components
        P = U[:, ::-1][:, 0:m]
        # PCA projection matrix
        DP = np.dot(P.T, D)
        if (m == 2):
            custom_scatter(0, 1, DP, L)
        m = m-1
    return


def computePCAsecondVersion(C, D, L):
    # Second version with overturned scatter plot
    # Compute SVD
    U, s, Vh = np.linalg.svd(C)
    # We can start from m=3
    m = D.shape[0]-1
    while(m > 0):
        # Principal components
        P = U[:, 0:m]
        # PCA projection matrix
        DP = np.dot(P.T, D)
        if (m == 2):
            custom_scatter(0, 1, DP, L)
        m = m-1
    return


def PCA(D, L):
    # Compute mean over columns of the dataset matrix (mean over columns means that
    # for the first row we get a value, for the second row we get a value, ecc.)
    mu = D.mean(axis=1)
    # Attention! mu is a 1-D array!
    # We want to subtract mean to all elements of the dataset (with broadcasting)
    # We need to reshape the 1-D array mu to a column vector 4x1
    mu = vcol(mu, mu.size)
    print(mu)
    # Now we can subtract (with broadcasting). C stands for centered
    DC = D - mu
    # Now we can compute the covariance matrix
    # DC.shape[1] is 150, it's the N parameter
    C = (1/DC.shape[1]) * (np.dot(DC, DC.T))
    print(C)
    computePCA(C, D, L)
    # computePCAsecondVersion(C,D,L)  #uncomment to test this version
    return


def computeBetweenClassCovarianceMatrix(D, L):
    # Compute mean over columns of the dataset matrix
    mu = D.mean(axis=1)
    # Reshape the 1-D array mu to a column vector 4x1
    mu = vcol(mu, mu.size)
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
    return (1/(n0+n1+n2))*((n0*np.dot(mu0-mu, (mu0-mu).T))+(n1*np.dot(mu1-mu, (mu1-mu).T)) +
                           (n2*np.dot(mu2-mu, (mu2-mu).T)))


def computeWithinClassCovarianceMatrix(D, L):
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
    # Compute within covariance matrix for each class
    Sw0 = (1/n0)*np.dot(D[:, L == 0]-mu0, (D[:, L == 0]-mu0).T)
    Sw1 = (1/n1)*np.dot(D[:, L == 1]-mu1, (D[:, L == 1]-mu1).T)
    Sw2 = (1/n2)*np.dot(D[:, L == 2]-mu2, (D[:, L == 2]-mu2).T)
    return (1/(n0+n1+n2))*(n0*Sw0+n1*Sw1+n2*Sw2)


def computeLDA(SB, SW, D, L):
    # Solve the generalized eigenvalue problem
    s, U = sc.eigh(SB, SW)
    # We can start from m=3
    m = U.shape[1]-1
    while(m > 0):
        # Compute W matrix from U
        W = U[:, ::-1][:, 0:m]
        # LDA projection matrix
        DP = np.dot(W.T, D)
        if (m == 2):
            custom_scatter(0, 1, DP, L)
        m = m-1
    return


def computeLDAsecondVersion(SB, SW, D, L):
    # Second version, more difficult and with overturned scatter plot
    # 3 classes => at most 2 directions. We start from m=2
    m = 2
    # Compute SVD
    U, s, _ = np.linalg.svd(SW)
    # Compute P1 matrix
    P1 = np.dot(np.dot(U, np.diag(1.0/(s**0.5))), U.T)
    # Compute transformed SB
    SBT = np.dot(np.dot(P1, SB), P1.T)
    # Get eigenvectors (columns of U ) from SBT
    _, U = np.linalg.eigh(SBT)
    while(m > 0):
        # Compute P2 (m leading eigenvectors)
        P2 = U[:, ::-1][:, 0:m]
        # Compute W
        W = np.dot(P1.T, P2)
        # LDA projection matrix
        DP = np.dot(W.T, D)
        if (m == 2):
            custom_scatter(0, 1, DP, L)
        m = m-1
    return


def LDA(D, L):
    SB = computeBetweenClassCovarianceMatrix(D, L)
    SW = computeWithinClassCovarianceMatrix(D, L)
    print(SB)
    print(SW)
    computeLDA(SB, SW, D, L)
    # computeLDAsecondVersion(SB, SW, D, L)   #uncomment to test this version


if __name__ == '__main__':
    D, L = load("iris.csv")
    PCA(D, L)
    LDA(D, L)
