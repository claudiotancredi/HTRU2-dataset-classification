# -*- coding: utf-8 -*-
"""
Created on Sun May 23 19:27:03 2021

Note: Functions are not so well written. This is due to the need of building
the solution step by step. They can be improved in the project, if necessary

@author: Claudio
"""

import numpy as np
import matplotlib.pyplot as plt


def plotROC(FPR, TPR):
    # Function used to plot TPR(FPR)
    plt.figure()
    plt.grid(linestyle='--')
    plt.plot(FPR, TPR, linewidth=2)
    return

def bayesErrorPlot(dcf, mindcf, effPriorLogOdds):
    plt.figure()
    plt.plot(effPriorLogOdds, dcf, label='DCF', color='r')
    plt.plot(effPriorLogOdds, mindcf, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    return


def confusionMatrix(pl, LEV, K, pi1, Cfn, Cfp):
    # Initialize matrix of K x K zeros
    matrix = np.zeros((K, K))
    # Here we're not talking about costs yet! We're only computing the confusion
    # matrix which "counts" how many times we get prediction i when the actual
    # label is j.
    for i in range(LEV.size):
        # Update "counter" in proper position
        matrix[pl[i], LEV[i]] += 1
    print("Confusion matrix with prior pi1=%.1f and costs Cfn=%.1f, Cfp=%.1f:" %
          (pi1, Cfn, Cfp))
    print(matrix)
    return matrix


def confusionMatrixV2THRESHOLD(pl, LEV, K, pi1, Cfn, Cfp):
    # Initialize matrix of K x K zeros
    matrix = np.zeros((K, K))
    # Here we're not talking about costs yet! We're only computing the confusion
    # matrix which "counts" how many times we get prediction i when the actual
    # label is j.
    for i in range(LEV.size):
        # Update "counter" in proper position
        matrix[pl[i], LEV[i]] += 1
    return matrix


def computeOptimalBayesDecisionBinaryTask(pi1, Cfn, Cfp, llrs, labels):
    # Compute the threshold
    t = -np.log((pi1*Cfn)/((1-pi1)*Cfp))
    # Now, if the llr is > than the threshold => predicted class is 1
    # If the llr is <= than the threshold => predicted class is 0
    predictedLabels = (llrs > t).astype(int)
    # Compute the confusion matrix
    m = confusionMatrix(predictedLabels, labels, 2, pi1, Cfn, Cfp)
    return m


def computeOptimalBayesDecisionBinaryTaskV2THRESHOLD(pi1, Cfn, Cfp, llrs, labels, t):
    # Now, if the llr is > than the threshold => predicted class is 1
    # If the llr is <= than the threshold => predicted class is 0
    predictedLabels = (llrs > t).astype(int)
    # Compute the confusion matrix
    m = confusionMatrixV2THRESHOLD(predictedLabels, labels, 2, pi1, Cfn, Cfp)
    return m


def evaluationBinaryTask(pi1, Cfn, Cfp, confMatrix):
    # Compute FNR and FPR
    FNR = confMatrix[0][1]/(confMatrix[0][1]+confMatrix[1][1])
    FPR = confMatrix[1][0]/(confMatrix[0][0]+confMatrix[1][0])
    # Compute empirical Bayes risk, that is the cost that we pay due to our
    # decisions c* for the test data.
    DCFu = pi1*Cfn*FNR+(1-pi1)*Cfp*FPR
    print("Unnormalized detection cost function with prior pi1=%.1f and costs Cfn=%.1f, Cfp=%.1f: %.3f" %
          (pi1, Cfn, Cfp, DCFu))
    return DCFu


def evaluationBinaryTaskV2THRESHOLD(pi1, Cfn, Cfp, confMatrix):
    # Compute FNR and FPR
    FNR = confMatrix[0][1]/(confMatrix[0][1]+confMatrix[1][1])
    FPR = confMatrix[1][0]/(confMatrix[0][0]+confMatrix[1][0])
    # Compute empirical Bayes risk, that is the cost that we pay due to our
    # decisions c* for the test data.
    DCFu = pi1*Cfn*FNR+(1-pi1)*Cfp*FPR
    return (DCFu, FNR, FPR)


def normalizedEvaluationBinaryTask(pi1, Cfn, Cfp, DCFu):
    # Define vector with dummy costs
    dummyCosts = np.array([pi1*Cfn, (1-pi1)*Cfp])
    # Compute risk for an optimal dummy system
    index = np.argmin(dummyCosts)
    # Compute normalized DCF
    DCFn = DCFu/dummyCosts[index]
    print("Normalized detection cost function with prior pi1=%.1f and costs Cfn=%.1f, Cfp=%.1f: %.3f" %
          (pi1, Cfn, Cfp, DCFn))
    return


def normalizedEvaluationBinaryTaskV2THRESHOLD(pi1, Cfn, Cfp, DCFu):
    # Define vector with dummy costs
    dummyCosts = np.array([pi1*Cfn, (1-pi1)*Cfp])
    # Compute risk for an optimal dummy system
    index = np.argmin(dummyCosts)
    # Compute normalized DCF
    DCFn = DCFu/dummyCosts[index]
    return DCFn


if __name__ == "__main__":
    # -------------------- BINARY TASK: OPTIMAL DECISIONS ----------------------------
    # Load log-likelihood ratios for the inferno-vs-paradiso task
    llrs = np.load("Data/commedia_llr_infpar.npy")
    # Load labels for this task. 1 is inferno, 0 is paradiso
    labels = np.load("Data/commedia_labels_infpar.npy")
    # Call the function to compute optimal Bayes decisions for different priors
    # and costs starting from binary llrs. The function receives the triplet
    # (pi1, Cfn, Cfp) where pi1 is the prior probability of class inferno,
    # Cfn is the cost of false negative, Cfp is the cost of false positive.
    # And of course it receives the llrs and the labels
    m1 = computeOptimalBayesDecisionBinaryTask(0.5, 1, 1, llrs, labels)
    m2 = computeOptimalBayesDecisionBinaryTask(0.8, 1, 1, llrs, labels)
    m3 = computeOptimalBayesDecisionBinaryTask(0.5, 10, 1, llrs, labels)
    m4 = computeOptimalBayesDecisionBinaryTask(0.8, 1, 10, llrs, labels)
    # Observations:
    # 1) When the prior for class 1 increases, the classifier tends to predict
    # class 1 more frequently.
    # 2) When the cost of predicting class 0 when the actual class is 1 increases
    # (Cfn), the classifier will make more false positive errors and less negative
    # errors to compensate the higher costs. The opposite happens when Cfp is higher.
    # -------------------- BINARY TASK: EVALUATION ----------------------------
    # We now turn our attention at evaluating the predictions through the Bayes
    # risk (unnormalized detection cost function)
    DCFu1 = evaluationBinaryTask(0.5, 1, 1, m1)
    DCFu2 = evaluationBinaryTask(0.8, 1, 1, m2)
    DCFu3 = evaluationBinaryTask(0.5, 10, 1, m3)
    DCFu4 = evaluationBinaryTask(0.8, 1, 10, m4)
    # Now we try the normalized detection cost function wrt the best dummy system
    normalizedEvaluationBinaryTask(0.5, 1, 1, DCFu1)
    normalizedEvaluationBinaryTask(0.8, 1, 1, DCFu2)
    normalizedEvaluationBinaryTask(0.5, 10, 1, DCFu3)
    normalizedEvaluationBinaryTask(0.8, 1, 10, DCFu4)
    # -------------------- MINIMUM DETECTION COSTS ----------------------------
    # We can compute the optimal threshold for a given application on the same
    # validation set that we're analyzing, and use such threshold for the test
    # population (K-fold cross validation can be also exploited to extract
    # validation sets from the training data when validation data is not available).
    # We can compute the normalized DCF over the test set using all possible
    # thresholds, and select its minimum value. This represents a lower bound
    # for the DCF that our system can achieve (minimum DCF).
    # To compute the minimum cost, we consider a set of threshold corresponding
    # to the set of test scores (llrs), sorted in increasing order. For each
    # threshold of this set, we compute the confusion matrix on the test set
    # itself and the corresponding normalized DCF using the code developed in
    # the previous section. The minimum DCF is the minimum of the obtained values.
    testScoresSorted = np.sort(llrs)
    # Define empty lists to store DCFs for the applications
    DCFarr1 = []
    DCFarr2 = []
    DCFarr3 = []
    DCFarr4 = []
    for t in testScoresSorted:
        # First application
        m1 = computeOptimalBayesDecisionBinaryTaskV2THRESHOLD(
            0.5, 1, 1, llrs, labels, t)
        (DCFu1, _, _) = evaluationBinaryTaskV2THRESHOLD(0.5, 1, 1, m1)
        DCFarr1.append(
            normalizedEvaluationBinaryTaskV2THRESHOLD(0.5, 1, 1, DCFu1))
        # Second application
        m2 = computeOptimalBayesDecisionBinaryTaskV2THRESHOLD(
            0.8, 1, 1, llrs, labels, t)
        (DCFu2, _, _) = evaluationBinaryTaskV2THRESHOLD(0.8, 1, 1, m2)
        DCFarr2.append(
            normalizedEvaluationBinaryTaskV2THRESHOLD(0.8, 1, 1, DCFu2))
        # Third application
        m3 = computeOptimalBayesDecisionBinaryTaskV2THRESHOLD(
            0.5, 10, 1, llrs, labels, t)
        (DCFu3, _, _) = evaluationBinaryTaskV2THRESHOLD(0.5, 10, 1, m3)
        DCFarr3.append(
            normalizedEvaluationBinaryTaskV2THRESHOLD(0.5, 10, 1, DCFu3))
        # Fourth application
        m4 = computeOptimalBayesDecisionBinaryTaskV2THRESHOLD(
            0.8, 1, 10, llrs, labels, t)
        (DCFu4, _, _) = evaluationBinaryTaskV2THRESHOLD(0.8, 1, 10, m4)
        DCFarr4.append(
            normalizedEvaluationBinaryTaskV2THRESHOLD(0.8, 1, 10, DCFu4))
    # Compute min of each array to get DCFn min for each application
    index = np.argmin(DCFarr1)
    print("Min DCF with prior pi1=%.1f and costs Cfn=%.1f, Cfp=%.1f: " %
          (0.5, 1, 1))
    print(DCFarr1[index])
    index = np.argmin(DCFarr2)
    print("Min DCF with prior pi1=%.1f and costs Cfn=%.1f, Cfp=%.1f: " %
          (0.8, 1, 1))
    print(DCFarr2[index])
    index = np.argmin(DCFarr3)
    print("Min DCF with prior pi1=%.1f and costs Cfn=%.1f, Cfp=%.1f: " %
          (0.5, 10, 1))
    print(DCFarr3[index])
    index = np.argmin(DCFarr4)
    print("Min DCF with prior pi1=%.1f and costs Cfn=%.1f, Cfp=%.1f: " %
          (0.8, 1, 10))
    print(DCFarr4[index])
    # By choosing a better threshold, we can observe that we've improved the
    # performances of the applications where the recognizer was harmful (>1).
    # These performances were affected by poor calibration.
    # ------------------------ ROC curves -----------------------------
    # ROC curves are used to plot false positive rates versus true positive rates
    # as the threshold varies.
    # The most commonly used ROC curves plot true positive rates TPRs against
    # false positive rates FPRs. We can compute TPRs from FNRs as TPR=1-FNR,
    # The ROC curve consists of points TPR(t), FPR(t).
    # Now, for each threshold we compute the confusion matrix and extract the
    # FNR and FPR as we did in the previous section. We compute TPR as said.
    # We plot the curve that contains on x-axis all the FPRs and on y-axis all
    # TPRs.
    # Define empty lists for x and y values
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3 = []
    y3 = []
    x4 = []
    y4 = []
    for t in testScoresSorted:
        # First application
        m1 = computeOptimalBayesDecisionBinaryTaskV2THRESHOLD(
            0.5, 1, 1, llrs, labels, t)
        (_, FNR1, FPR1) = evaluationBinaryTaskV2THRESHOLD(0.5, 1, 1, m1)
        TPR1 = 1-FNR1
        x1.append(FPR1)
        y1.append(TPR1)
        # Second application
        m2 = computeOptimalBayesDecisionBinaryTaskV2THRESHOLD(
            0.8, 1, 1, llrs, labels, t)
        (_, FNR2, FPR2) = evaluationBinaryTaskV2THRESHOLD(0.8, 1, 1, m2)
        TPR2 = 1-FNR2
        x2.append(FPR2)
        y2.append(TPR2)
        # Third application
        m3 = computeOptimalBayesDecisionBinaryTaskV2THRESHOLD(
            0.5, 10, 1, llrs, labels, t)
        (_, FNR3, FPR3) = evaluationBinaryTaskV2THRESHOLD(0.8, 1, 1, m3)
        TPR3 = 1-FNR3
        x3.append(FPR3)
        y3.append(TPR3)
        # Fourth application
        m4 = computeOptimalBayesDecisionBinaryTaskV2THRESHOLD(
            0.8, 1, 10, llrs, labels, t)
        (_, FNR4, FPR4) = evaluationBinaryTaskV2THRESHOLD(0.8, 1, 1, m4)
        TPR4 = 1-FNR4
        x4.append(FPR4)
        y4.append(TPR4)
    plotROC(x1, y1)
    plotROC(x2, y2)
    plotROC(x3, y3)
    plotROC(x4, y4)
    # The plots are all the same, this is because we actually don't consider
    # prior and costs. They are passed to functions but are not used.
    # -------------------- Bayes error plots --------------------------
    # The last tool that we consider to assess the performance of our recognizer
    # consists in plotting the normalized costs as a function of an effective
    # prior pitilde. We have seen that, for binary problems, an application
    # (pi1, Cfn, Cfp) is indeed equivalent to an application (pitilde, 1, 1).
    # So we can represent different applications just by varying the value of
    # pitilde.
    # The normalized Bayes error plot allows assessing the performance of the
    # recognizer AS WE VARY THE APPLICATION, as a function of prior log-odds
    # ptilde (which are the negative of the theoretical threshold for the
    # considered application).
    # We now compute the Bayes error plot for our recognizer. We consider 21 values
    # of ptilde ranging, for example, from -3 to +3. For each ptilde we can compute
    # the corresponding effective prior just by inverting the relation of ptilde.
    # Then we compute the normalized DCF and the normalized minimum DCF corresponding
    # to pitilde. Then we plot the computed values as a function of log-odds
    # ptilde: the x-axis contains the values of ptilde, the y-axis the corresponding
    # DCF.
    effPriorLogOdds = np.linspace(-3, 3, 21)
    effPriors = 1/(1+np.exp(-1*effPriorLogOdds))
    DCFnarr = []
    minDCFn = []
    for i in range(21):
        #temp list to store all the DCF for the various thresholds
        temp=[]
        m = computeOptimalBayesDecisionBinaryTaskV2THRESHOLD(
            effPriors[i], 1, 1, llrs, labels, -np.log((effPriors[i])/((1-effPriors[i]))))
        (DCFu, _, _) = evaluationBinaryTaskV2THRESHOLD(effPriors[i], 1, 1, m)
        # Append the normalized DCF to the list
        DCFnarr.append(
            normalizedEvaluationBinaryTaskV2THRESHOLD(effPriors[i], 1, 1, DCFu))
        for t in testScoresSorted:
            m1 = computeOptimalBayesDecisionBinaryTaskV2THRESHOLD(
                effPriors[i], 1, 1, llrs, labels, t)
            (DCFu1, _, _) = evaluationBinaryTaskV2THRESHOLD(effPriors[i], 1, 1, m1)
            # Append the normalized DCF to the temp list
            temp.append(
                normalizedEvaluationBinaryTaskV2THRESHOLD(effPriors[i], 1, 1, DCFu1))
        # Find index of the min value
        index = np.argmin(temp)
        # Append the min normalized DCF to the corresponding list
        minDCFn.append(temp[index])
    # Call the function to plot
    bayesErrorPlot(DCFnarr, minDCFn, effPriorLogOdds)