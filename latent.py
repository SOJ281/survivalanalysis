import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sksurv.datasets import load_breast_cancer

from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder

from sklearn import set_config
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import time

import pyreadstat
import math 
from lca import *
from stepmix.stepmix import StepMix
from stepmix.utils import get_mixed_descriptor


set_config(display="text")  # displays text representation of estimators


print("Starting")

#df = read_spss('MG.sav')
Xs, meta = pyreadstat.read_sav('./MG.sav')
#ClinicAge, MaleSex, Haemoglobin, Diabetes
#X = X.loc[:, X.columns != "ClinicAge" and X.columns != "MaleSex" and X.columns != "Haemoglobin" and X.columns != "Diabetes" ]
#X = Xs.loc[:, Xs.columns != ["ClinicAge", "MaleSex", "Haemoglobin", "Diabetes"] ]
X = Xs.loc[:, ~Xs.columns.isin(["ClinicAge", "MaleSex", "Haemoglobin", "Diabetes", "TimeDeath", "StatusDeath"]) == False]
X['ClinicAge'] = X['ClinicAge'].div(5)
#X['ClinicAge'] = X['ClinicAge'].round(0)
#X['ClinicAge'] = X['ClinicAge']/5
X = X.dropna()
X.reset_index(drop=True, inplace=True)


ypd = Xs.loc[:, ~Xs.columns.isin(["TimeDeath", "StatusDeath"]) == False]
y = np.empty(dtype=[('X.StatusDeath', bool), ('X.TimeDeath', np.float64)], shape=(len(X.index)))
#y['X.StatusDeath'] = (X['StatusDeath'] == 1).values
y['X.StatusDeath'] = X['StatusDeath'].values
y['X.TimeDeath'] = X['TimeDeath'].values
#print(type(X))
print("Participants", len((X).index))
print("MaleSex", len((X.loc[X['MaleSex'] == 1.0]).index))
print("StatusDeath", len((X).index)-len((X.loc[X['StatusDeath'] == 1.0]).index))
print("Diabetes", len((X.loc[X['Diabetes'] == 1.0]).index))
print("meanAge", X['ClinicAge'].mean())
print("meanHaemoglobin", X['Haemoglobin'].mean())
print("medianTimeDeath", X['TimeDeath'].median()/365)

del X["StatusDeath"]
del X["TimeDeath"]
#del X["DeathDate"]

print("X",X)
print("X",X.shape)
print("y",y.shape)





Xt = X

mixed_data, mixed_descriptor = get_mixed_descriptor(
    dataframe=Xt,
    binary=['MaleSex','Diabetes'],
    continuous=['ClinicAge', 'Haemoglobin']
)



procedure = 3
#bestClass = [1, 2000000]
bestClass = []
for c in range(1, 8):
    model = StepMix(n_components=c, measurement=mixed_descriptor, verbose=1, max_iter=10000)
    model.fit(mixed_data)
    bic = model.bic(mixed_data)
    #print("Class ",c," BIC:", bic)
    #if (bestClass[1] > bic):
        #bestClass = [c, bic]
    bestClass.append([c, bic])
    #bestClass = [c, bic]

best = 1
for x in bestClass:
    print("Class ",x[0]," BIC:", x[1])
    if (x[1] < bestClass[best-1][1]):
        best = x[0]
        
print("Best class count is ",best)
        
#model = StepMix(n_components=bestClass[best][0], measurement=mixed_descriptor, verbose=1, max_iter=10000)
model = StepMix(n_components=best, measurement=mixed_descriptor, verbose=1, max_iter=10000)
model.fit(mixed_data)
classProba = model.predict_proba(mixed_data)
mixed_data['mixed_pred'] = model.predict(mixed_data)
#print(mixed_data)
#print(mixed_data['mixed_pred'].value_counts())

columns=[]

for i in range(best):
    columns.append("Class_"+str(i+1))

#print("columns",columns)
latentFrame = pd.DataFrame(classProba, columns=columns).round(2)

if (procedure== 2):
    pyreadstat.write_sav(pd.concat([Xt, ypd], axis = 1).dropna(), "latentP2.sav")
    Xt = latentFrame.dropna()
else:
    Xt = pd.concat([Xt, latentFrame], axis = 1)
    pyreadstat.write_sav(pd.concat([Xt, ypd], axis = 1).dropna(), "latentP3.sav")
    Xt = Xt.dropna()

#print("Xt",Xt.shape)
#print("y",y.shape)


    
#Normal
import math
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import scipy.stats as st
start = time.time()
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
coefficients = {}

cph = CoxPHSurvivalAnalysis()
cv_resultsa = cross_validate(cph, Xt, y, cv=10)
print(cv_resultsa)
print("CI median", np.median(cv_resultsa["test_score"]))
ciMean = np.mean(cv_resultsa["test_score"])
print("CI Mean", ciMean, st.t.interval(alpha=0.95, df=len(cv_resultsa["test_score"])-1,loc=ciMean,scale=st.sem(cv_resultsa["test_score"])))

cv_results = cross_validate(cph, Xt, y, cv=10, return_estimator=True)
print(cv_results['estimator'][0].feature_names_in_)
print("len(cv_results['estimator'])",len(cv_results['estimator']))
coefficients = []
for model in cv_results['estimator']:
    coefficients.append(model.coef_.tolist())
coefficientsMean = np.mean(coefficients, axis=0)
print("CoeffMean", coefficientsMean, st.t.interval(alpha=0.95, df=len(coefficients)-1,loc=coefficientsMean,scale=st.sem(coefficients)))

hazRatio = np.exp(coefficients)
hazRatioMean = np.mean(hazRatio, axis=0)
print("hazard ratio", hazRatioMean, st.t.interval(alpha=0.95, df=len(hazRatio)-1,loc=hazRatioMean,scale=st.sem(hazRatio)))


end = time.time()
print("Normal done in ", (end - start),"s")    