#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:29:46 2022

@author: ashrith
"""
"""
Higgs Data set:  Train 25000 x 33
                 Test  55000 x 31 (no weight and label column)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib as plt
import scipy
import os
import inspect
import sys
import hyperopt

from hyperopt import hp, fmin, tpe, Trials
from xgboost import plot_tree
from sklearn import tree
from functools import partial
from skopt import space
from skopt import gp_minimize

# main only execs if this is the source file
if __name__ == "__main__":
    tests_df = pd.read_csv('/home/ashrith/github/Higgs_XGboost/dataset/test.csv')
    train_df = pd.read_csv('/home/ashrith/github/Higgs_XGboost/dataset/training.csv')

# By dropping first we don't get 2 different columns for Label, rather one just for s (1 if s and 0 if b)
train_df = pd.get_dummies(train_df, columns=['Label'], drop_first=True)

ncols_train = train_df.shape[1]
ncols_tests = tests_df.shape[1]
Y_train  = train_df.iloc[:,ncols_train-1]
X_train  = train_df.iloc[:,1:ncols_train-2]
Y_tests  = tests_df.iloc[:,ncols_tests-1]
X_tests  = tests_df.iloc[:,1:ncols_tests-2]

weight = train_df.iloc[:,ncols_train-2] * len(Y_tests) / len(Y_train)
sum_wpos = sum( weight[i] for i in range(len(Y_train)) if Y_train[i] == 1 ) #b and s
sum_wneg = sum( weight[i] for i in range(len(Y_train)) if Y_train[i] == 0 )

# Param list
param = {}
param['objective'] = 'binary:logistic'
# scale weight of positive examples
param['scale_pos_weight'] = sum_wneg/sum_wpos
param['eta'] = 0.15
#param['lambda']=0.1 (part of denominator in formula)
#param['gamma']=0.1 (if gain > gamma, then keep leaf)
#param['subsample'] = 0.5
#param['min_child_weight']= 0 (sum of cover should be less than this for leaf to exist)
#param['colsample_bytree']= 1 (0-1] range
#param['max_depth'] = 6
param['eval_metric'] = 'auc'
param['silent'] = 1
param['nthread'] = 1
#param['early_stopping_rounds']=10
#param['seed'] = seed
#random.seed(seed)
evals=list([('eval_metric','auc')]+[('eval_metric', 'ams@0.15')])
param_list = list(param.items())+[('eval_metric', 'ams@0.15')]
eval_list=[(evals)]
num_round = 360
print ('loading data end, start to boost trees')

#Model: Dmatrix
D_train = xgb.DMatrix( X_train, Y_train, missing = -999.0, weight=weight )
D_tests = xgb.DMatrix(X_tests, Y_tests)
watchlist_train = [ (D_train,'train') ]
watchlist_tests =  [ (D_tests,'tests') ]

#this works
boosted_tree = xgb.train( param_list, D_train, num_round, watchlist_train, early_stopping_rounds=15,verbose_eval=10);

""" 
#Model: XGBClassifier
clf = xgb.XGBClassifier(objective='binary:logistic',missing=-999.0, weight=weight, seed=42)
#clf.fit(X_train, Y_train, verbose=True, early_stopping_rounds=10, eval_metric='auc', eval_set=[(X_tests, Y_tests)])

"""
"""
#Hyperopt:
param_space = {
                "max_depth": hp.quniform("max_depth", 3, 15, 1),
                "n_estimators": hp.quniform("n_estimators", 100, 600, 1),
                "criterion": hp.choice("criterion", ["gini", "entropy"]),
                "max_features": hp.uniform("max_features", 0.01, 1),
                }

optimization_function = partial(optimize, x=X, y=Y)
trials = Trials()

result = fmin(
             fn = optimization_function,
             space = param_space,
             algo = tpe.suggest,
             max_evals = 15,
             trials = trials,
             )

print (result)
"""