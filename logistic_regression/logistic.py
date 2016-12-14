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
    X, y = load_svmlight_file(dir_path + "/" + y1)
    model.fit(X, y)
    X, y = load_svmlight_file(dir_path + "/" + y2)
    yp = model.predict(X)
    return np.sum(np.abs(y - yp))/float(y.shape[0])

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


