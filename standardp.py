import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline



from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder

from sklearn import set_config
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import time

import pyreadstat
import math 

set_config(display="text")  # displays text representation of estimators


print("Starting")
Xs, meta = pyreadstat.read_sav('./MG.sav')
#ClinicAge, MaleSex, Haemoglobin, Diabetes
X = Xs.loc[:, ~Xs.columns.isin(["ClinicAge", "MaleSex", "Haemoglobin", "Diabetes", "TimeDeath", "StatusDeath"]) == False]
X['ClinicAge'] = X['ClinicAge'].div(5)
X = X.dropna()

y = np.empty(dtype=[('X.StatusDeath', bool), ('X.TimeDeath', np.float64)], shape=(len(X.index)))
y['X.StatusDeath'] = X['StatusDeath'].values
y['X.TimeDeath'] = X['TimeDeath'].values
print("Participants", len((X).index))
print("MaleSex", len((X.loc[X['MaleSex'] == 1.0]).index))
print("StatusDeath", len((X).index)-len((X.loc[X['StatusDeath'] == 1.0]).index))
print("Diabetes", len((X.loc[X['Diabetes'] == 1.0]).index))
print("meanAge", X['ClinicAge'].mean())
print("meanHaemoglobin", X['Haemoglobin'].mean())
print("medianTimeDeath", X['TimeDeath'].median()/365)

del X["StatusDeath"]
del X["TimeDeath"]


Xt = X




    
#Normal
import math
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import scipy.stats as st
start = time.time()
coefficients = {}

cph = CoxPHSurvivalAnalysis()
cv_resultsa = cross_validate(cph, Xt, y, cv=10)
print(cv_resultsa)
print("CI median", np.median(cv_resultsa["test_score"]), st.t.interval(confidence=0.95, df=len(cv_resultsa["test_score"])-1,loc=np.median(cv_resultsa["test_score"]),scale=st.sem(cv_resultsa["test_score"])))

cv_results = cross_validate(cph, Xt, y, cv=10, return_estimator=True)
print(cv_results['estimator'][0].feature_names_in_)
print("len(cv_results['estimator'])",len(cv_results['estimator']))
coefficients = []
for model in cv_results['estimator']:
    coefficients.append(model.coef_.tolist())
    
coeffMedian = np.median(coefficients, axis=0)
print("CoeffMedian", coeffMedian, st.t.interval(confidence=0.95, df=len(coefficients)-1,loc=coeffMedian,scale=st.sem(coefficients)))

hazRatio = np.exp(coefficients)
hazRatioMedian = np.median(hazRatio, axis=0)
print("hazard ratio", hazRatioMedian, st.t.interval(confidence=0.95, df=len(hazRatio)-1,loc=hazRatioMedian,scale=st.sem(hazRatio)))



end = time.time()
print("Normal done in ", (end - start),"s")    