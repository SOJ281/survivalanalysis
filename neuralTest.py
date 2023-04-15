from cox_nnet import *
import numpy
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

# load data
import pandas as pd
import numpy as np



#from sklearn import set_config
#from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

#from netCode import *
import sklearn

import time

import pyreadstat
import math 

import random
#set_config(display="text")  # displays text representation of estimators


print("Starting")
#df = read_spss('MG.sav')
Xs, meta = pyreadstat.read_sav('./MG.sav')
#ClinicAge, MaleSex, Haemoglobin, Diabetes
#X = X.loc[:, X.columns != "ClinicAge" and X.columns != "MaleSex" and X.columns != "Haemoglobin" and X.columns != "Diabetes" ]
#X = Xs.loc[:, Xs.columns != ["ClinicAge", "MaleSex", "Haemoglobin", "Diabetes"] ]
X = Xs.loc[:, ~Xs.columns.isin(["ClinicAge", "MaleSex", "Haemoglobin", "Diabetes", "TimeDeath", "StatusDeath"]) == False]
X['ClinicAge'] = X['ClinicAge'].div(5)
X = X.dropna()



#y = np.empty(dtype=[('X.StatusDeath', bool), ('X.TimeDeath', np.float64)], shape=(len(X.index)))
ytime = np.empty(dtype=[ ('X.TimeDeath', np.float64)], shape=(len(X.index)))
#y['X.StatusDeath'] = X['StatusDeath'].values
#ytime['X.TimeDeath'] = X['TimeDeath'].values
ytime =  X["TimeDeath"]
yStatus = X["StatusDeath"]

print("MaleSex", len((X.loc[X['MaleSex'] == 1.0]).index))
print("StatusDeath", len((X.loc[X['StatusDeath'] == 1.0]).index))
print("Diabetes", len((X.loc[X['Diabetes'] == 1.0]).index))
print("meanAge", X['ClinicAge'].mean())
print("meanHaemoglobin", X['Haemoglobin'].mean())
print("medianTimeDeath", X['TimeDeath'].median()/365)

del X["StatusDeath"]
del X["TimeDeath"]


num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('std_scaler',StandardScaler())
])

#x = X.to_numpy()
x = num_pipeline.fit_transform(X)
ystatus = yStatus.to_numpy()
ytime = ytime.to_numpy()


print("xtype", type(x))

# split into test/train sets
x_train, x_test, ytime_train, ytime_test, ystatus_train, ystatus_test = \
    train_test_split(x, ytime, ystatus, test_size = 0.2, random_state = 100)

print("x_train", type(x_train))
print("ystatus_train", type(ystatus_train))


#Define parameters
final_results = []
l2_params = numpy.asarray(numpy.arange(-6, -2, .5))
l2_params = numpy.asarray(numpy.linspace(-5, -2., 4))
#l2_params = numpy.asarray([np.exp(i*i)*i for i in numpy.linspace(-2, 2, 4)])
#l2_params = numpy.asarray(numpy.linspace(-3.25, -2.75, 2))
#l2_params = numpy.asarray(numpy.linspace(-4, -2, 2))
#max_iterations = [750, 1000]
max_iterations = [500, 1000, 1500]
learning_rate = [0.1]
layer_count = [1, 3, 5, 9]
node_count = [1, 3, 5, 8]
weight_count = [1]
#learning_rate = [0.1]
print("l2_params",l2_params)
print("max_iterations",max_iterations)



import itertools
import scipy.stats as st

#model_params = dict(node_map = None, input_split = None, L2_reg = l2_params)
model_params = dict(node_map = [[(4,[0],5), (4,[0],5), (4,[0],5)]], input_split = None, L2_reg = l2_params)

counter = 1
searches = itertools.product(max_iterations, learning_rate, layer_count)

for (max_iter, lr, lc) in itertools.product(max_iterations, learning_rate, layer_count):
    search_params = dict(method = "nesterov", max_iter=max_iter, learning_rate=lr)

    prior_weights = x.shape[1]
    if lc > 1:
        #print("NEW", new_weights)
        nd_map = []
        for i in range(lc):
            new_weights =random.randint(1,9)
            nd_map.append([(prior_weights,[0],new_weights)]*random.randint(1,9))
            prior_weights = new_weights
    else:
        nd_map = [[(x.shape[1],[0],random.randint(1,9))]*random.randint(1,6)]

    model_params = dict(node_map = nd_map, input_split = None, L2_reg = l2_params)



    print("Search:", counter,"/", len(list(searches)), "with", nd_map)
    print("max_iter:", max_iter,"/",lr, lc)
 
    cv_params = dict(cv_seed=1, n_folds=5, cv_metric = "cindex", L2_range = l2_params, search_iters = 5)
    counter+=1

    #bestIndex,bestL2,bestCVPL=L2CVSearch(x_train,ytime_train,ystatus_train,
    #    model_params = model_params,search_params = search_params,cv_params = cv_params, verbose=False)

    #final_results.append(dict(search_params=search_params,cv_params=cv_params,bestL2=bestL2,bestCVPL = bestCVPL, bestIndex = bestIndex))

    Cindex, L2_reg_params, mean_cvpl = L2CVProfile(x,ytime,ystatus,#x_train,ytime_train,ystatus_train,
	    model_params,search_params,cv_params, verbose=False)
    new_val = dict(search_params=search_params,cv_params=cv_params,L2_reg_params=L2_reg_params, mean_cvpl = mean_cvpl, Cindex= Cindex, node_map = nd_map)
    for i in range(len(new_val["mean_cvpl"])):
        print("L2",new_val["L2_reg_params"][i],"mean_cvpl",new_val["mean_cvpl"][i],"CI interval",st.t.interval(alpha=0.95, df=len(new_val["Cindex"][i])-1,
              loc=new_val["mean_cvpl"][i],
              scale=st.sem(new_val["Cindex"][i])))
    #print(dict(search_params=search_params,cv_params=cv_params,L2_reg_params=L2_reg_params, mean_cvpl = mean_cvpl, node_map = nd_map))
    final_results.append(new_val)
    
print("FINISHED")
print(final_results)


exit(0)



