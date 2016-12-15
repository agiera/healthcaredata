import sklearn
import scipy
from scipy.sparse import csr_matrix
from sklearn import cross_validation
from sklearn.model_selection import KFold
from sklearn.datasets import load_svmlight_file
import numpy as np
from src.TTRegression import *





def cv_exp_machine(file, kfolds=5, ratio=0.2):
    X, y = load_svmlight_file(file)
    X, y = sklearn.utils.shuffle(X, y)
    
    # even out pos and negs
    # scipy isn't working well with shorter code
    index = []
    if np.sum(y) > X.shape[0]/2:
        n = np.sum(y) - X.shape[0]/2
        mask = np.ones(y.shape[0])
        for i in range(X.shape[0]):
            if n == 0:
                break
            if y[i] == 1:
                index.append(i)
                n -= 1
    elif np.sum(y) < X.shape[0]/2:
        n = y.shape[0]/2 - np.sum(y)
        mask = np.ones(X.shape[0])
        for i in range(X.shape[0]):
            if n == 0:
                break
            if y[i] == 0:
                index.append(i)
                n -= 1
    mask = np.delete(np.array(range(X.shape[0])), index)
    X = X[mask, :]
    y = y[mask]
    
    cv = cross_validation.KFold(X.shape[0], n_folds=kfolds)
    mses = []
    for traincv, testcv in cv:
        X_tr = X[traincv, :]
        X_te = X[testcv, :]
        
        y_tr = y[traincv]
        y_te = y[testcv]
        
        model = TTRegression('all-subsets', 'logistic', rank=4, solver='riemannian-sgd', max_iter=2000, verbose=2)
        model.fit(X_tr.toarray(), y_tr)
        probs = model.predict_proba(X_te.toarray())
        yp = np.array([np.argmax(prob) for prob in probs])
        mses.append(np.dot(y_te - yp, y_te - yp)/y_te.shape[0])
    return mses
    
def predict_y(f1, f2):
    X, y = load_svmlight_file(f1)
    model = TTRegression('all-subsets', 'logistic', rank=4, solver='riemannian-sgd', max_iter=1000, verbose=2)
    model.fit(X.toarray(), y)
    X, y = load_svmlight_file(f2)
    probs = model.predict_proba(X.toarray())
    yp = np.array([np.argmax(prob) for prob in probs])
    return np.dot(y - yp, y - yp)/y.shape[0]


with open("results.txt", 'a') as f:
    f.write("2011 training 2010: "+str(predict_y("exp_4_2010.libsvm", "exp_4_2011.libsvm")) + "\n")
with open("results.txt", 'a') as f:
    f.write("2012 training 2011: "+str(predict_y("exp_4_2011.libsvm", "exp_4_2012.libsvm")) + "\n")
with open("results.txt", 'a') as f:
    f.write("2013 training 2012: "+str(predict_y("exp_4_2012.libsvm", "exp_4_2013.libsvm")) + "\n")
with open("results.txt", 'a') as f:
    f.write("2014 training 2013: "+str(predict_y("exp_4_2013.libsvm", "exp_4_2014.libsvm")) + "\n")



