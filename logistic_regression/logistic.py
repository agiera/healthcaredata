from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import cross_val_score
import numpy as np
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))


#predicted = cross_validation.cross_val_predict(LogisticRegression(), iris['data'], iris['target'], cv=10)


def cv_logit(file, kfolds=5):
    X, y = load_svmlight_file(file)
    return cross_val_score(LogisticRegression(),
                      X,
                      y, 
                      fit_params={}, 
                      cv = kfolds, 
                      scoring = 'mean_squared_error')
                      
                      
def pred_next_years(y1, y2):
    model = LogisticRegression()
    X_tr, y_tr = load_svmlight_file(dir_path + "/" + y1)
    X_te, y_te = load_svmlight_file(dir_path + "/" + y2)
    if X_tr.shape[1] > X_te.shape[1]:
        X_te = np.pad(X_te.toarray(), ((0, 0),(0, X_tr.shape[1] - X_te.shape[1])), mode='constant', constant_values=0)
    elif X_tr.shape[1] < X_te.shape[1]:
        X_tr = np.pad(X_tr.toarray(), ((0, 0),(0, X_te.shape[1] - X_tr.shape[1])), mode='constant', constant_values=0)
    model.fit(X_tr, y_tr)
    yp = model.predict(X_te)
    return np.sum(np.abs(y_te - yp))/float(y_te.shape[0])

#print("meps_2010_sf_sso.libfm: " + str(cv_logit("meps_2010_sf_sso.libfm")) + "\n")
#print("meps_2011_sf_sso.libfm: " + str(cv_logit("meps_2011_sf_sso.libfm")) + "\n")
#print("meps_2012_sf_sso.libfm: " + str(cv_logit("meps_2012_sf_sso.libfm")) + "\n")
#print("meps_2013_sf_sso.libfm: " + str(cv_logit("meps_2013_sf_sso.libfm")) + "\n")
#print("meps_2014_sf_sso.libfm: " + str(cv_logit("meps_2014_sf_sso.libfm")) + "\n")

# X, y = load_svmlight_file(dir_path + "/meps_2010_cccf_sso.libsvm")
# print(np.sum(y))
# X, y = load_svmlight_file(dir_path + "/meps_2010_cccf_sso.libsvm")
# print(np.sum(y))
# X, y = load_svmlight_file(dir_path + "/meps_2010_cccf_sso.libsvm")
# print(np.sum(y))
# X, y = load_svmlight_file(dir_path + "/meps_2010_cccf_sso.libsvm")
# print(np.sum(y))
# X, y = load_svmlight_file(dir_path + "/meps_2010_cccf_sso.libsvm")
# print(np.sum(y))

print("pred 2011: " + str(pred_next_years("meps_2010_cccf_sso.libsvm", "meps_2011_cccf_sso.libsvm")))
print("pred 2012: " + str(pred_next_years("meps_2011_cccf_sso.libsvm", "meps_2012_cccf_sso.libsvm")))
print("pred 2013: " + str(pred_next_years("meps_2012_cccf_sso.libsvm", "meps_2013_cccf_sso.libsvm")))
print("pred 2014: " + str(pred_next_years("meps_2013_cccf_sso.libsvm", "meps_2014_cccf_sso.libsvm")))



#for i in range(4):
    #print("pred 201"+str(i+1)+": " + str(pred_next_years("fourth experiment/exp_4_201"+str(i)+".libsvm", "fourth experiment/exp_4_201"+str(i+1)+".libsvm")))

