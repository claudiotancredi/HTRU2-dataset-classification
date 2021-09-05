# -*- coding: utf-8 -*-
"""
Created on Sun May  9 23:22:26 2021

@author: Claudio

The script is taken from the professor's solution. 
My changes are at lines: 309-320, 366-371
"""

import numpy
import string
import scipy.special
import itertools
import sys


def mcol(v):
    return v.reshape((v.size, 1))


def load_data():

    lInf = []

    if sys.version_info.major == 3:  # Check if Python version is Python 3 or Python 2
        f = open('data/inferno.txt', encoding="ISO-8859-1")
    else:
        f = open('data/inferno.txt')

    for line in f:
        lInf.append(line.strip())
    f.close()

    lPur = []

    if sys.version_info.major == 3:  # Check if Python version is Python 3 or Python 2
        f = open('data/purgatorio.txt', encoding="ISO-8859-1")
    else:
        f = open('data/purgatorio.txt')

    for line in f:
        lPur.append(line.strip())
    f.close()

    lPar = []

    if sys.version_info.major == 3:  # Check if Python version is Python 3 or Python 2
        f = open('data/paradiso.txt', encoding="ISO-8859-1")
    else:
        f = open('data/paradiso.txt')
    for line in f:
        lPar.append(line.strip())
    f.close()

    return lInf, lPur, lPar


def split_data(l, n):

    lTrain, lTest = [], []
    for i in range(len(l)):
        if i % n == 0:
            lTest.append(l[i])
        else:
            lTrain.append(l[i])

    return lTrain, lTest

### Solution 1 - Dictionaries of frequencies ###


def S1_buildDictionary(lTercets):
    '''
    Create a set of all words contained in the list of tercets lTercets
    lTercets is a list of tercets (list of strings)
    '''

    sDict = set([])
    for s in lTercets:
        words = s.split()
        for w in words:
            sDict.add(w)
    return sDict


def S1_estimateModel(hlTercets, eps=0.1):
    '''
    Build frequency dictionaries for each class.

    hlTercets: dict whose keys are the classes, and the values are the list of tercets of each class.
    eps: smoothing factor (pseudo-count)

    Return: dictionary h_clsLogProb whose keys are the classes. For each class, h_clsLogProb[cls] is a dictionary whose keys are words and values are the corresponding log-frequencies (model parameters for class cls)
    '''

    # Build the set of all words appearing at least once in each classes
    sDictCommon = set([])

    for cls in hlTercets:  # Loop over class labels
        lTercets = hlTercets[cls]
        sDictCls = S1_buildDictionary(lTercets)
        sDictCommon = sDictCommon.union(sDictCls)

    # Initialize the counts of words for each class with eps
    h_clsLogProb = {}
    for cls in hlTercets:  # Loop over class labels
        # Create a dictionary for each class that contains all words as keys and the pseudo-count as initial values
        h_clsLogProb[cls] = {w: eps for w in sDictCommon}

    # Estimate counts
    for cls in hlTercets:  # Loop over class labels
        lTercets = hlTercets[cls]
        for tercet in lTercets:  # Loop over all tercets of the class
            words = tercet.split()
            for w in words:  # Loop over words of the given tercet
                h_clsLogProb[cls][w] += 1

    # Compute frequencies
    for cls in hlTercets:  # Loop over class labels
        # Get all occurrencies of words in cls and sum them. this is the number of words (including pseudo-counts)
        nWordsCls = sum(h_clsLogProb[cls].values())
        for w in h_clsLogProb[cls]:  # Loop over all words
            h_clsLogProb[cls][w] = numpy.log(
                h_clsLogProb[cls][w]) - numpy.log(nWordsCls)  # Compute log N_{cls,w} / N

    return h_clsLogProb


def S1_compute_logLikelihoods(h_clsLogProb, text):
    '''
    Compute the array of log-likelihoods for each class for the given text
    h_clsLogProb is the dictionary of model parameters as returned by S1_estimateModel
    The function returns a dictionary of class-conditional log-likelihoods
    '''

    logLikelihoodCls = {cls: 0 for cls in h_clsLogProb}
    for cls in h_clsLogProb:  # Loop over classes
        for word in text.split():  # Loop over words
            if word in h_clsLogProb[cls]:
                logLikelihoodCls[cls] += h_clsLogProb[cls][word]
    return logLikelihoodCls


def S1_compute_logLikelihoodMatrix(h_clsLogProb, lTercets, hCls2Idx=None):
    '''
    Compute the matrix of class-conditional log-likelihoods for each class each tercet in lTercets

    h_clsLogProb is the dictionary of model parameters as returned by S1_estimateModel
    lTercets is a list of tercets (list of strings)
    hCls2Idx: map between textual labels (keys of h_clsLogProb) and matrix rows. If not provided, automatic mapping based on alphabetical oreder is used

    Returns a #cls x #tercets matrix. Each row corresponds to a class.
    '''

    if hCls2Idx is None:
        hCls2Idx = {cls: idx for idx, cls in enumerate(sorted(h_clsLogProb))}

    S = numpy.zeros((len(h_clsLogProb), len(lTercets)))
    for tIdx, tercet in enumerate(lTercets):
        hScores = S1_compute_logLikelihoods(h_clsLogProb, tercet)
        for cls in h_clsLogProb:  # We sort the class labels so that rows are ordered according to alphabetical order of labels
            clsIdx = hCls2Idx[cls]
            S[clsIdx, tIdx] = hScores[cls]

    return S

### Solution 2 - Arrays of occurrencies ###


def S2_buildDictionary(lTercets):
    '''
    Create a dictionary of all words contained in the list of tercets lTercets
    The dictionary allows storing the words, and mapping each word to an index i (the corresponding index in the array of occurrencies)

    lTercets is a list of tercets (list of strings)
    '''

    hDict = {}
    nWords = 0
    for tercet in lTercets:
        words = tercet.split()
        for w in words:
            if w not in hDict:
                hDict[w] = nWords
                nWords += 1
    return hDict


def S2_estimateModel(hlTercets, eps=0.1):
    '''
    Build word log-probability vectors for all classes

    hlTercets: dict whose keys are the classes, and the values are the list of tercets of each class.
    eps: smoothing factor (pseudo-count)

    Return: tuple (h_clsLogProb, h_wordDict). h_clsLogProb is a dictionary whose keys are the classes. For each class, h_clsLogProb[cls] is an array containing, in position i, the log-frequency of the word whose index is i. h_wordDict is a dictionary that maps each word to its corresponding index.
    '''

    # Since the dictionary also includes mappings from word to indices it's more practical to build a single dict directly from the complete set of tercets, rather than doing it incrementally as we did in Solution S1
    lTercetsAll = list(itertools.chain(*hlTercets.values()))
    hWordDict = S2_buildDictionary(lTercetsAll)
    nWords = len(hWordDict)  # Total number of words

    h_clsLogProb = {}
    for cls in hlTercets:
        # In this case we use 1-dimensional vectors for the model parameters. We will reshape them later.
        h_clsLogProb[cls] = numpy.zeros(nWords) + eps

    # Estimate counts
    for cls in hlTercets:  # Loop over class labels
        lTercets = hlTercets[cls]
        for tercet in lTercets:  # Loop over all tercets of the class
            words = tercet.split()
            for w in words:  # Loop over words of the given tercet
                wordIdx = hWordDict[w]
                # h_clsLogProb[cls] ius a 1-D array, h_clsLogProb[cls][wordIdx] is the element in position wordIdx
                h_clsLogProb[cls][wordIdx] += 1

    # Compute frequencies
    for cls in h_clsLogProb.keys():  # Loop over class labels
        vOccurrencies = h_clsLogProb[cls]
        vFrequencies = vOccurrencies / vOccurrencies.sum()
        vLogProbabilities = numpy.log(vFrequencies)
        h_clsLogProb[cls] = vLogProbabilities

    return h_clsLogProb, hWordDict


def S2_tercet2occurrencies(tercet, hWordDict):
    '''
    Convert a tercet in a (column) vector of word occurrencies. Word indices are given by hWordDict
    '''
    v = numpy.zeros(len(hWordDict))
    for w in tercet.split():
        if w in hWordDict:  # We discard words that are not in the dictionary
            v[hWordDict[w]] += 1
    return mcol(v)


def S2_compute_logLikelihoodMatrix(h_clsLogProb, hWordDict, lTercets, hCls2Idx=None):
    '''
    Compute the matrix of class-conditional log-likelihoods for each class each tercet in lTercets

    h_clsLogProb and hWordDict are the dictionary of model parameters and word indices as returned by S2_estimateModel
    lTercets is a list of tercets (list of strings)
    hCls2Idx: map between textual labels (keys of h_clsLogProb) and matrix rows. If not provided, automatic mapping based on alphabetical oreder is used

    Returns a #cls x #tercets matrix. Each row corresponds to a class.
    '''

    if hCls2Idx is None:
        hCls2Idx = {cls: idx for idx, cls in enumerate(sorted(h_clsLogProb))}

    numClasses = len(h_clsLogProb)
    numWords = len(hWordDict)

    # We build the matrix of model parameters. Each row contains the model parameters for a class (the row index is given from hCls2Idx)
    MParameters = numpy.zeros((numClasses, numWords))
    for cls in h_clsLogProb:
        clsIdx = hCls2Idx[cls]
        # MParameters[clsIdx, :] is a 1-dimensional view that corresponds to the row clsIdx, we can assign to the row directly the values of another 1-dimensional array
        MParameters[clsIdx, :] = h_clsLogProb[cls]

    SList = []
    for tercet in lTercets:
        v = S2_tercet2occurrencies(tercet, hWordDict)
        # The log-lieklihoods for the tercets can be computed as a matrix-vector product. Each row of the resulting column vector corresponds to M_c v = sum_j v_j log p_c,j
        STercet = numpy.dot(MParameters, v)
        SList.append(numpy.dot(MParameters, v))

    S = numpy.hstack(SList)
    return S


################################################################################

def compute_classPosteriors(S, logPrior=None):
    '''
    Compute class posterior probabilities

    S: Matrix of class-conditional log-likelihoods
    logPrior: array with class prior probability (shape (#cls, ) or (#cls, 1)). If None, uniform priors will be used

    Returns: matrix of class posterior probabilities
    '''

    if logPrior is None:
        logPrior = numpy.log(numpy.ones(S.shape[0]) / float(S.shape[0]))
    J = S + mcol(logPrior)  # Compute joint probability
    # Compute marginal likelihood log f(x)
    ll = scipy.special.logsumexp(J, axis=0)
    # Compute posterior log-probabilities P = log ( f(x, c) / f(x)) = log f(x, c) - log f(x)
    P = J - ll
    return numpy.exp(P)


def compute_accuracy(P, L):
    '''
    Compute accuracy for posterior probabilities P and labels L. L is the integer associated to the correct label (in alphabetical order)
    '''

    PredictedLabel = numpy.argmax(P, axis=0)
    NCorrect = (PredictedLabel.ravel() == L.ravel()).sum()
    NTotal = L.size
    return float(NCorrect)/float(NTotal)


def confusionMatrix(pl, LEV, K):
    # Initialize matrix of K x K zeros
    matrix = numpy.zeros((K, K))
    # Here we're not talking about costs yet! We're only computing the confusion
    # matrix which "counts" how many times we get prediction i when the actual
    # label is j.
    for i in range(LEV.size):
        # Update "counter" in proper position
        matrix[pl[i], LEV[i]] += 1
    print("Confusion matrix:")
    print(matrix)
    return


if __name__ == '__main__':

    lInf, lPur, lPar = load_data()

    lInfTrain, lInfEval = split_data(lInf, 4)
    lPurTrain, lPurEval = split_data(lPur, 4)
    lParTrain, lParEval = split_data(lPar, 4)

    ### Solution 1 ###
    ### Multiclass ###

    hCls2Idx = {'inferno': 0, 'purgatorio': 1, 'paradiso': 2}

    hlTercetsTrain = {
        'inferno': lInfTrain,
        'purgatorio': lPurTrain,
        'paradiso': lParTrain
    }

    lTercetsEval = lInfEval + lPurEval + lParEval

    S1_model = S1_estimateModel(hlTercetsTrain, eps=0.001)

    S1_predictions = compute_classPosteriors(
        S1_compute_logLikelihoodMatrix(
            S1_model,
            lTercetsEval,
            hCls2Idx,
        ),
        numpy.log(numpy.array([1./3., 1./3., 1./3.]))
    )

    labelsInf = numpy.zeros(len(lInfEval))
    labelsInf[:] = hCls2Idx['inferno']

    labelsPar = numpy.zeros(len(lParEval))
    labelsPar[:] = hCls2Idx['paradiso']

    labelsPur = numpy.zeros(len(lPurEval))
    labelsPur[:] = hCls2Idx['purgatorio']

    labelsEval = numpy.hstack([labelsInf, labelsPur, labelsPar])

    # S1_predictions is the matrix with class posterior probabilities,
    # then we compute the predicted labels with argmax. labelsEval includes
    # the labels of the evaluation set, but it's a vector of float, so we can
    # cast it to int.
    confusionMatrix(numpy.argmax(S1_predictions, axis=0),
                    labelsEval.astype(numpy.int), int(labelsEval.max()) + 1)

    # Per-class accuracy
    print('Multiclass - S1 - Inferno - Accuracy:', compute_accuracy(
        S1_predictions[:, labelsEval == hCls2Idx['inferno']], labelsEval[labelsEval == hCls2Idx['inferno']]))
    print('Multiclass - S1 - Purgatorio - Accuracy:', compute_accuracy(
        S1_predictions[:, labelsEval == hCls2Idx['purgatorio']], labelsEval[labelsEval == hCls2Idx['purgatorio']]))
    print('Multiclass - S1 - Paradiso - Accuracy:', compute_accuracy(
        S1_predictions[:, labelsEval == hCls2Idx['paradiso']], labelsEval[labelsEval == hCls2Idx['paradiso']]))

    # Overall accuracy
    print('Multiclass - S1 - Accuracy:',
          compute_accuracy(S1_predictions, labelsEval))

    ### Binary from multiclass scores [Optional, for the standard binary case see below] ###
    ### Only inferno vs paradiso, the other pairs are similar ###

    lTercetsEval = lInfEval + lParEval
    S = S1_compute_logLikelihoodMatrix(
        S1_model, lTercetsEval, hCls2Idx=hCls2Idx)

    SBinary = numpy.vstack([S[0:1, :], S[2:3, :]])
    P = compute_classPosteriors(SBinary)
    labelsEval = numpy.hstack([labelsInf, labelsPar])
    # Since labelsPar == 2, but the row of Paradiso in SBinary has become row 1 (row 0 is Inferno), we have to modify the labels for paradise, otherwise the function compute_accuracy will not work
    labelsEval[labelsEval == 2] = 1

    print('Binary (From multiclass) - S1 - Accuracy:',
          compute_accuracy(P, labelsEval))

    ### Binary ###
    ### Only inferno vs paradiso, the other pairs are similar ###

    hCls2Idx = {'inferno': 0, 'paradiso': 1}

    hlTercetsTrain = {
        'inferno': lInfTrain,
        'paradiso': lParTrain
    }

    lTercetsEval = lInfEval + lParEval

    S1_model = S1_estimateModel(hlTercetsTrain, eps=0.001)

    S1_predictions = compute_classPosteriors(
        S1_compute_logLikelihoodMatrix(
            S1_model,
            lTercetsEval,
            hCls2Idx,
        ),
        numpy.log(numpy.array([1./2., 1./2.]))
    )

    labelsInf = numpy.zeros(len(lInfEval))
    labelsInf[:] = hCls2Idx['inferno']

    labelsPar = numpy.zeros(len(lParEval))
    labelsPar[:] = hCls2Idx['paradiso']

    labelsEval = numpy.hstack([labelsInf, labelsPar])

    print('Binary [inferno vs paradiso] - S1 - Accuracy:',
          compute_accuracy(S1_predictions, labelsEval))

    ### Solution 2 ###
    ### Multiclass ###

    hCls2Idx = {'inferno': 0, 'purgatorio': 1, 'paradiso': 2}

    hlTercetsTrain = {
        'inferno': lInfTrain,
        'purgatorio': lPurTrain,
        'paradiso': lParTrain
    }

    lTercetsEval = lInfEval + lPurEval + lParEval

    S2_model, S2_wordDict = S2_estimateModel(hlTercetsTrain, eps=0.001)

    S2_predictions = compute_classPosteriors(
        S2_compute_logLikelihoodMatrix(
            S2_model,
            S2_wordDict,
            lTercetsEval,
            hCls2Idx,
        ),
        numpy.log(numpy.array([1./3., 1./3., 1./3.]))
    )

    labelsInf = numpy.zeros(len(lInfEval))
    labelsInf[:] = hCls2Idx['inferno']

    labelsPar = numpy.zeros(len(lParEval))
    labelsPar[:] = hCls2Idx['paradiso']

    labelsPur = numpy.zeros(len(lPurEval))
    labelsPur[:] = hCls2Idx['purgatorio']

    labelsEval = numpy.hstack([labelsInf, labelsPur, labelsPar])

    print('Multiclass - S2 - Accuracy:',
          compute_accuracy(S2_predictions, labelsEval))

    ### Binary from multiclass scores [Optional, for the standard binary case see below] ###
    ### Only inferno vs paradiso, the other pairs are similar ###

    lTercetsEval = lInfEval + lParEval
    S = S2_compute_logLikelihoodMatrix(
        S2_model, S2_wordDict, lTercetsEval, hCls2Idx=hCls2Idx)

    SBinary = numpy.vstack([S[0:1, :], S[2:3, :]])
    P = compute_classPosteriors(SBinary)
    labelsEval = numpy.hstack([labelsInf, labelsPar])
    # Since labelsPar == 2, but the row of Paradiso in SBinary has become row 1 (row 0 is Inferno), we have to modify the labels for paradise, otherwise the function compute_accuracy will not work
    labelsEval[labelsEval == 2] = 1

    print('Binary (From multiclass) - S2 - Accuracy:',
          compute_accuracy(P, labelsEval))

    ### Binary ###
    ### Only inferno vs paradiso, the other pairs are similar ###

    hCls2Idx = {'inferno': 0, 'paradiso': 1}

    hlTercetsTrain = {
        'inferno': lInfTrain,
        'paradiso': lParTrain
    }

    lTercetsEval = lInfEval + lParEval

    S2_model, S2_wordDict = S2_estimateModel(hlTercetsTrain, eps=0.001)

    S2_predictions = compute_classPosteriors(
        S2_compute_logLikelihoodMatrix(
            S2_model,
            S2_wordDict,
            lTercetsEval,
            hCls2Idx,
        ),
        numpy.log(numpy.array([1./2., 1./2.]))
    )

    labelsInf = numpy.zeros(len(lInfEval))
    labelsInf[:] = hCls2Idx['inferno']

    labelsPar = numpy.zeros(len(lParEval))
    labelsPar[:] = hCls2Idx['paradiso']

    labelsEval = numpy.hstack([labelsInf, labelsPar])

    print('Binary [inferno vs paradiso] - S2 - Accuracy:',
          compute_accuracy(S2_predictions, labelsEval))
