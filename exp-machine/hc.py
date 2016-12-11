import sklearn
import scipy
from scipy.sparse import csr_matrix
from sklearn import cross_validation
from sklearn.model_selection import KFold
from sklearn.datasets import load_svmlight_file
import numpy as np
from src.TTRegression import *





def cv_exp_machine(file, kfolds=5, ratio=0.1):
    X, y = load_svmlight_file(file)
    X, y = sklearn.utils.shuffle(X, y)
    
    # even out pos and negs
    # scipy isn't working well with shorter code
    if np.sum(y) > X.shape[0]/2:
        n = np.sum(y) - X.shape[0]/2
        mask = np.ones(y.shape[0])
        for i in range(X.shape[0]):
            if n == 0:
                break
            if y[i] == 1:
                mask[i] = 0
                n -= 1
    else:
        n = y.shape[0]/2 - np.sum(y)
        mask = np.ones(X.shape[0])
        for i in range(X.shape[0]):
            if n == 0:
                break
            if y[i] == 0:
                mask[i] = 0
                n -= 1
    X = X[mask, :]
    y = np.logical_and(y, mask)
    
    cv = cross_validation.KFold(X.shape[0], n_folds=kfolds)
    mses = []
    for traincv, testcv in cv:
        mask = np.zeros(X.shape[0])
        mask[traincv] = 1
        X_tr = X[mask, :]
        mask = np.zeros(X.shape[0])
        mask[testcv] = 1
        X_te = X[np.logical_not(mask), :]
        
        y_tr = y[traincv]
        y_te = y[testcv]
        
        model = TTRegression('all-subsets', 'logistic', rank=4, solver='riemannian-sgd', max_iter=10, verbose=2)
        model.fit(X_tr, y_tr)
        yp = model.decision_function(X_te)
        mses.append(np.norm(y_te - yp))
        model.destroy()
    return mses



print("meps_2010_sf_sso.libfm: " + str(cv_exp_machine("meps_2010_sf_sso.libfm")) + "\n")
print("meps_2011_sf_sso.libfm: " + str(cv_exp_machine("meps_2011_sf_sso.libfm")) + "\n")
print("meps_2012_sf_sso.libfm: " + str(cv_exp_machine("meps_2012_sf_sso.libfm")) + "\n")
print("meps_2013_sf_sso.libfm: " + str(cv_exp_machine("meps_2013_sf_sso.libfm")) + "\n")
print("meps_2014_sf_sso.libfm: " + str(cv_exp_machine("meps_2014_sf_sso.libfm")) + "\n")


