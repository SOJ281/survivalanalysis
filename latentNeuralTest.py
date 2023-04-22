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
if 5 == input("Enter number for procedure 5 or 6"):
    print("Proc 5:")
    classes, meta = pyreadstat.read_sav('./latentP2.sav')
else:
    classes, meta = pyreadstat.read_sav('./latentP3.sav')
X = classes.loc[:, ~classes.columns.isin(["TimeDeath", "StatusDeath"]) != False]
X = X.dropna()

print("X", X)
ypd = classes.loc[:, ~classes.columns.isin(["TimeDeath", "StatusDeath"]) == False]
ypd = ypd.dropna()
ytime = np.empty(dtype=[ ('ypd.TimeDeath', np.float64)], shape=(len(X.index)))
ytime =  ypd["TimeDeath"]
yStatus = ypd["StatusDeath"]


num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('std_scaler',StandardScaler())
])

#x = X.to_numpy()
x = num_pipeline.fit_transform(X)
ystatus = yStatus.to_numpy()
ytime = ytime.to_numpy()
print(x)



#Define parameters
final_results = []
l2_params = numpy.asarray(numpy.linspace(-5, -2., 4))

max_iterations = [500]
learning_rate = [0.1]
layer_count = [1, 3, 5, 9]
node_count = [1, 3, 5, 8]
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
            node_count = random.randint(1,9)
            nd_map.append([(prior_weights,[0],new_weights)]*node_count)
            prior_weights = new_weights
            print(node_count,"|",new_weights)
    else:
        nd_map = [[(x.shape[1],[0],random.randint(1,9))]*random.randint(1,9)]

    model_params = dict(node_map = nd_map, input_split = None, L2_reg = l2_params)



    print("Search:", counter,"/", len(list(searches)), "with", nd_map)
    print("max_iter:", max_iter,"/",lr, lc)
 
    cv_params = dict(cv_seed=1, n_folds=10, cv_metric = "cindex", L2_range = l2_params, search_iters = 5)
    counter+=1


    Cindex, L2_reg_params, mean_cvpl = L2CVProfile(x,ytime,ystatus,
	    model_params,search_params,cv_params, verbose=False)
    new_val = dict(search_params=search_params,cv_params=cv_params,L2_reg_params=L2_reg_params, mean_cvpl = mean_cvpl, Cindex= Cindex, node_map = nd_map)
    for i in range(len(new_val["mean_cvpl"])):
        print("L2",new_val["L2_reg_params"][i],"mean_cvpl",new_val["mean_cvpl"][i],"CI interval",st.t.interval(alpha=0.95, df=len(new_val["Cindex"][i])-1,
              loc=new_val["mean_cvpl"][i],
              scale=st.sem(new_val["Cindex"][i])))
    #print(dict(search_params=search_params,cv_params=cv_params,L2_reg_params=L2_reg_params, mean_cvpl = mean_cvpl, node_map = nd_map))
    final_results.append(new_val)
    
print("FINISHED")




