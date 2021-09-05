# -*- coding: utf-8 -*-
"""

@authors: Claudio Tancredi, Francesca Russo
"""

import numpy as np

def Jgradrebalanced(w, b, DTR, LTR, lambd, prior):
    normTerm = lambd/2*(np.linalg.norm(w)**2)
    sumTermTrueClass = 0
    sumTermFalseClass = 0
    for i in range(DTR.shape[1]):
        argexpnegative = -np.dot(w.T, DTR[:, i])-b
        flagArgExpNegative = False
        argexppositive = np.dot(w.T, DTR[:, i])+b
        flagArgExpPositive = False
        if (argexpnegative>709):
            flagArgExpNegative=True
        if (argexppositive>709):
            flagArgExpPositive=True
        if LTR[i]==1:
            if (flagArgExpNegative==True):
                sumTermTrueClass += argexpnegative
            else:
                sumTermTrueClass += np.log1p(np.exp(-np.dot(w.T, DTR[:, i])-b))
        else:
            if (flagArgExpPositive==True):
                sumTermFalseClass+=argexppositive
            else:
                sumTermFalseClass += np.log1p(np.exp(np.dot(w.T, DTR[:, i])+b))
    j = normTerm + (prior/DTR[:, LTR==1].shape[1])*sumTermTrueClass + ((1-prior)/DTR[:, LTR==0].shape[1])*sumTermFalseClass
    return j

# def Jgrad(w, b, DTR, LTR, lambd):
#     # The computation of log(1+x) can lead to numerical issues when x is small,
#     # since the sum will make the contribution of x disappear. We can avoid the
#     # issue using function np.log1p which computes log(1+x) in a numerically more
#     # stable way. ATTENTION, the term 1+ is already included, we just need to pass
#     # x as argument.
#     normTerm = lambd/2*(np.linalg.norm(w)**2)
#     sumTerm = 0
#     for i in range(DTR.shape[1]):
#         argexpnegative = -np.dot(w.T, DTR[:, i])-b
#         flagArgExpNegative = False
#         argexppositive = np.dot(w.T, DTR[:, i])+b
#         flagArgExpPositive = False
#         if (argexpnegative>709):
#             flagArgExpNegative=True
#         if (argexppositive>709):
#             flagArgExpPositive=True
#         if (flagArgExpNegative==True and flagArgExpPositive == True):
#             sumTerm += LTR[i] * argexpnegative + \
#                         (1-LTR[i])*argexppositive
#         elif (flagArgExpNegative==False and flagArgExpPositive==True):
#             sumTerm += LTR[i] * np.log1p(np.exp(argexpnegative)) + \
#                         (1-LTR[i])*argexppositive
#         elif (flagArgExpNegative==True and flagArgExpPositive==False):
#             sumTerm += LTR[i] * argexpnegative + \
#                         (1-LTR[i])*np.log1p(np.exp(argexppositive))
#         else:
#             sumTerm += LTR[i] * np.log1p(np.exp(-np.dot(w.T, DTR[:, i])-b)) + \
#                 (1-LTR[i])*np.log1p(np.exp(np.dot(w.T, DTR[:, i])+b))
#     j = normTerm + (1/DTR.shape[1])*sumTerm
#     sumTerm=0
#     for i in range(DTR.shape[1]):
#         argexpnegative = -np.dot(w.T, DTR[:, i])-b
#         flagArgExpNegative = False
#         argexppositive = np.dot(w.T, DTR[:, i])+b
#         flagArgExpPositive = False
#         if (argexpnegative>709):
#             flagArgExpNegative=True
#         if (argexppositive>709):
#             flagArgExpPositive=True
#         if (flagArgExpNegative==True and flagArgExpPositive == True):
#             sumTerm+=LTR[i]*(-DTR[:, i])+(1-LTR[i])*(DTR[:, i])
#         elif (flagArgExpNegative==False and flagArgExpPositive==True):
#             sumTerm+=LTR[i]*(1/(1+np.exp(-np.dot(w.T, DTR[:, i])-b)))*np.exp(-np.dot(w.T, DTR[:, i])-b)*\
#                 (-DTR[:, i])+(1-LTR[i])*(DTR[:, i])
#         elif (flagArgExpNegative==True and flagArgExpPositive==False):
#             sumTerm+=LTR[i]*(-DTR[:, i])+(1-LTR[i])*(1/(1+np.exp(np.dot(w.T, DTR[:, i])+b)))*np.exp(np.dot(w.T, DTR[:, i])+b)*\
#                     (DTR[:, i])
#         else:
#             sumTerm+=LTR[i]*(1/(1+np.exp(-np.dot(w.T, DTR[:, i])-b)))*np.exp(-np.dot(w.T, DTR[:, i])-b)*\
#                 (-DTR[:, i])+(1-LTR[i])*(1/(1+np.exp(np.dot(w.T, DTR[:, i])+b)))*np.exp(np.dot(w.T, DTR[:, i])+b)*\
#                     (DTR[:, i])
#     derw=lambd*w+ (1/DTR.shape[1]) * sumTerm
#     sumTerm=0
#     for i in range(DTR.shape[1]):
#         argexpnegative = -np.dot(w.T, DTR[:, i])-b
#         flagArgExpNegative = False
#         argexppositive = np.dot(w.T, DTR[:, i])+b
#         flagArgExpPositive = False
#         if (argexpnegative>709):
#             flagArgExpNegative=True
#         if (argexppositive>709):
#             flagArgExpPositive=True
#         if (flagArgExpNegative==True and flagArgExpPositive == True):
#             sumTerm+=LTR[i]*(-1)+(1-LTR[i])
#         elif (flagArgExpNegative==False and flagArgExpPositive==True):
#             sumTerm+=LTR[i]*(1/(1+np.exp(-np.dot(w.T, DTR[:, i])-b)))*np.exp(-np.dot(w.T, DTR[:, i])-b)*\
#                 (-1)+(1-LTR[i])
#         elif (flagArgExpNegative==True and flagArgExpPositive==False):
#             sumTerm+=LTR[i]*\
#                 (-1)+(1-LTR[i])*(1/(1+np.exp(np.dot(w.T, DTR[:, i])+b)))*np.exp(np.dot(w.T, DTR[:, i])+b)
#         else:
#             sumTerm+=LTR[i]*(1/(1+np.exp(-np.dot(w.T, DTR[:, i])-b)))*np.exp(-np.dot(w.T, DTR[:, i])-b)*\
#                 (-1)+(1-LTR[i])*(1/(1+np.exp(np.dot(w.T, DTR[:, i])+b)))*np.exp(np.dot(w.T, DTR[:, i])+b)
#     derb=(1/DTR.shape[1]) * sumTerm
#     return (j,  np.hstack((derw, derb)))

# def J(w, b, DTR, LTR, lambd):
#     # The computation of log(1+x) can lead to numerical issues when x is small,
#     # since the sum will make the contribution of x disappear. We can avoid the
#     # issue using function np.log1p which computes log(1+x) in a numerically more
#     # stable way. ATTENTION, the term 1+ is already included, we just need to pass
#     # x as argument.
#     normTerm = lambd/2*(np.linalg.norm(w)**2)
#     sumTerm = 0
#     for i in range(DTR.shape[1]):
#         argexpnegative = -np.dot(w.T, DTR[:, i])-b
#         flagArgExpNegative = False
#         argexppositive = np.dot(w.T, DTR[:, i])+b
#         flagArgExpPositive = False
#         if (argexpnegative>709):
#             flagArgExpNegative=True
#         if (argexppositive>709):
#             flagArgExpPositive=True
#         if (flagArgExpNegative==True and flagArgExpPositive == True):
#             sumTerm += LTR[i] * argexpnegative + \
#                         (1-LTR[i])*argexppositive
#         elif (flagArgExpNegative==False and flagArgExpPositive==True):
#             sumTerm += LTR[i] * np.log1p(np.exp(argexpnegative)) + \
#                         (1-LTR[i])*argexppositive
#         elif (flagArgExpNegative==True and flagArgExpPositive==False):
#             sumTerm += LTR[i] * argexpnegative + \
#                         (1-LTR[i])*np.log1p(np.exp(argexppositive))
#         else:
#             sumTerm += LTR[i] * np.log1p(np.exp(-np.dot(w.T, DTR[:, i])-b)) + \
#                 (1-LTR[i])*np.log1p(np.exp(np.dot(w.T, DTR[:, i])+b))
#     return normTerm + (1/DTR.shape[1])*sumTerm

def logreg_obj(v, DTR, LTR, l, prior=0.5):
    # This function should receive a single numpy array v with shape (D+1),
    # where D is the dimensionality of the feature space (example, D=4 for IRIS,
    # 4 features). v should pack all model parameters (w and b), then we can unpack
    # the array here.
    w, b = v[0:-1], v[-1]
    # The function has to access also DTR, LTR and lambda, which are required
    # to compute the objective. We can choose different strategies, but I
    # choose the easiest way.
    j = Jgradrebalanced(w, b, DTR, LTR, l, prior)
    return j
