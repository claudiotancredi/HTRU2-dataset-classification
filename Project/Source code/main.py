# -*- coding: utf-8 -*-
"""

@author: Claudio
"""
import utils
import PCA
import numpy as np
import matplotlib.pyplot as plt
import GaussianClassifierTiedCov
import LogisticRegression

def plotClassFeatures(i, j, xlabel, ylabel, D0, D1, classesNames):
    plt.figure()
    plt.scatter(D0[0,:], D0[1,:], color="#1e90ff", s=10)
    plt.scatter(D1[0,:], D1[1,:], color="#ff8c00", s=10)
    plt.legend(classesNames)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    return

if __name__=="__main__":
    D, L = utils.load("Train.txt")
    TD, TL = utils.load("Test.txt")
    ZD, _, _ =utils.ZNormalization(D)
    PCA7 = PCA.PCA(ZD, L, 7)
    ZTD, _, _ = utils.ZNormalization(TD)
    PCA7TD = PCA.PCA(ZTD, L, 7)
    gc = GaussianClassifierTiedCov.GaussianClassifierTiedCov()
    lr = LogisticRegression.LogisticRegression()
    gc.train(PCA7, L)
    print(utils.computeErrorRate(gc.predict(PCA7TD), TL))
    lr.train(PCA7, L, 1e-4, 0.5)
    print(utils.computeErrorRate(lr.predict(PCA7TD), TL))
    # PCA2= PCA.PCA(ZD, L, 2)
    # # PLOT PCA 2
    # utils.custom_scatter(0, 1, "1st component", "2nd component", PCA2, L, utils.classesNames)
    # # CENTER REDUCED DATA FOR EACH CLASS
    # PCA2centered0 = utils.centerData(PCA2[:, L==0])
    # PCA2centered1 = utils.centerData(PCA2[:, L==1])
    # initialcov0=np.cov(PCA2centered0)
    # initialcov1=np.cov(PCA2centered1)
    # initialsigma = 1/(D.shape[1])*(D[:, L == 0].shape[1]*initialcov0+D[:, L == 1].shape[1]*initialcov1)
    # print(initialsigma-initialcov0)
    # print(initialsigma-initialcov1)
    # utils.plotClassFeatures(0, 1, "1st component", "2nd component", PCA2centered0, PCA2centered1, utils.classesNames)
    # #test
    # eigenvalues1, eigenvectors1 = np.linalg.eigh(initialcov1)
    # eigenvalues0, eigenvectors0 = np.linalg.eigh(initialcov0)
    # eigenvaluesSigma, eigenvectorsSigma = np.linalg.eigh(initialsigma)
    # theta = np.linspace(0, 2*np.pi, 1000);
    # ellipsis0 = (np.sqrt(eigenvalues0[None,:]) * eigenvectors0) @ [np.sin(theta), np.cos(theta)] *3
    # ellipsis1 = (np.sqrt(eigenvalues1[None,:]) * eigenvectors1) @ [np.sin(theta), np.cos(theta)]*3
    # ellipsisSigma = (np.sqrt(eigenvaluesSigma[None,:]) * eigenvectorsSigma) @ [np.sin(theta), np.cos(theta)]*3
    # plt.figure()
    # plt.scatter(PCA2centered0[0,:], PCA2centered0[1,:], color="#1e90ff", s=10)
    # plt.scatter(PCA2centered1[0,:], PCA2centered1[1,:], color="#ff8c00", s=10)
    # plt.legend(utils.classesNames)
    # plt.ylabel("2nd component")
    # plt.xlabel("1st component")
    # plt.plot(ellipsis0[0,:], ellipsis0[1,:], color="b")
    # plt.plot(ellipsis1[0,:], ellipsis1[1,:], color="r")
    # plt.plot(ellipsisSigma[0,:], ellipsisSigma[1,:], color="g")
    # # OVERTURN CLASS 1 MATRIX TO OBTAIN AN OVERTURNED PLOT
    # PCA2centered1[0,:] = -PCA2centered1[0,:]
    # utils.plotClassFeatures(0, 1, "1st component", "2nd component", PCA2centered0, PCA2centered1, utils.classesNames)
    # cov1prerotation=np.cov(PCA2centered1)
    # # APPLY ROTATION TO ALIGN DATA OF THE TWO CLASSES
    # theta = np.pi/6
    # m_rotation = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    # PCA2centered1 = np.dot(m_rotation,PCA2centered1)
    # # COMPUTE EIGENVALUES AND EIGENVECTORS TO PLOT ELLIPSIS
    # cov1=np.cov(PCA2centered1)
    # eigenvalues1, eigenvectors1 = np.linalg.eigh(cov1)
    # cov0=np.cov(PCA2centered0)
    # eigenvalues0, eigenvectors0 = np.linalg.eigh(cov0)
    # sigma = 1/(D.shape[1])*(D[:, L == 0].shape[1]*cov0+D[:, L == 1].shape[1]*cov1)
    # print(sigma-cov0)
    # print(sigma-cov1)
    # # print(initialcov1)
    # # print(cov1prerotation)
    # # print(cov1)
    # theta = np.linspace(0, 2*np.pi, 1000);
    # ellipsis0 = (np.sqrt(eigenvalues0[None,:]) * eigenvectors0) @ [np.sin(theta), np.cos(theta)] *4
    # ellipsis1 = (np.sqrt(eigenvalues1[None,:]) * eigenvectors1) @ [np.sin(theta), np.cos(theta)]*2
    # plt.figure()
    # plt.scatter(PCA2centered0[0,:], PCA2centered0[1,:], color="#1e90ff", s=10)
    # plt.scatter(PCA2centered1[0,:], PCA2centered1[1,:], color="#ff8c00", s=10)
    # plt.legend(utils.classesNames)
    # plt.ylabel("2nd component")
    # plt.xlabel("1st component")
    # plt.plot(ellipsis0[0,:], ellipsis0[1,:], color="b")
    # plt.plot(ellipsis1[0,:], ellipsis1[1,:], color="r")
    
    #EXPLAINED VARIANCE
    # ev = []
    # ev.append(np.sum(np.linalg.eigh(np.cov(ZD))[0]))
    # for i in range(6):
    #     PCAi = PCA.PCA(ZD, L, 8-i-1)
    #     ev.append(np.sum(np.linalg.eigh(np.cov(PCAi))[0]))
    # ref = ev[0]
    # for i in range(len(ev)):
    #     ev[i] = ev[i]/ref
    # ev.reverse()
    # plt.figure()
    # plt.plot([2,3,4,5,6,7,8], ev, label='act DCF', color='b')
    # plt.xlabel("# of components")
    # plt.ylabel("Explained variance")
    
    # PCA2 = PCA.PCA(ZD, L, 1)
    # utils.plotFeatures(PCA2, L, ["1st component"], utils.classesNames)