import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.svm import OneClassSVM
from matplotlib.pyplot import figure

def SVMOneClass(X):
    clf = OneClassSVM(gamma=0.025,nu=0.8)
    fit = clf.fit(X)
    return fit
def get_Support_Vectors(fitted_model):
    return fitted_model.support_vectors_

def get_DIs(support_vectors,matrix):
    mu = np.mean(support_vectors,axis=0)
    cov_inv = np.linalg.inv(np.cov(support_vectors.T))
    DI = []
    for i in range(len(matrix)):
        temp = np.log((((matrix[i]-mu).T).dot(cov_inv)).dot(matrix[i]-mu)) #mahalanobis distance
        DI.append(temp)
    DI = np.array(DI)
    return DI
def getConfidenceInterval(DI):
    return [-2*np.std(DI) + np.mean(DI),2*np.std(DI) + np.mean(DI)]

def concat(DI_undamaged,DI_damaged):
    return np.concatenate((DI_undamaged,DI_damaged))

def plot_characteristics(DI,interval):
    figure(num=None, figsize=(10,10), dpi=80, facecolor='w', edgecolor='k')
    y = [i for i in range(len(DI))]
    plt.scatter(y,DI)
    plt.hlines(interval[0],0,850,'r')
    plt.hlines(interval[1],0,850,'r')
    plt.xlabel("Test Number")
    plt.ylabel("log(DI)")
    plt.vlines(450,2,10)
    i = 500
    while i<=850:
        plt.vlines(i,2,10,'g')
        i+=50
    plt.show()
 def error(DI_undamaged,DI_damaged,interval):