import numpy as np
import sklearn
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import KFold
from sklearn.datasets import load_svmlight_file
from src.TTRegression import *




def get_pos_neg(x, y):
    pos_idx = []
    neg_idx = []
    for i in range(len(x)):
        if y[i] == 1:
            pos_idx.append(x[i])
        else:
            neg_idx.append(x[i])
    return pos_idx, neg_idx


def cv_exp_machine(file, kfolds=5):
    X, y = load_svmlight_file(file)
    return cross_val_score(EXPRegressor(max_iter=10),
                      X,
                      y, 
                      fit_params={'l':0.1}, 
                      cv = kfolds, 
                      scoring = 'mean_squared_error')

class EXPRegressor:
    def __init__(self, max_iter=10):
        self.model = TTRegression('all-subsets', 'logistic', rank=4, solver='riemannian-sgd', max_iter=max_iter, verbose=2)

    def predict(self, X):
        return self.model.predict_proba(X)

    def classify(self, inputs):
        return self.model.decision_function(inputs)

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)

    def get_params(self, deep = False):
        #return {'max_iter':self.l}
        return {}





















