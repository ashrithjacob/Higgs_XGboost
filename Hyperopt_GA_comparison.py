"""
Created on Wed Jun 29 15:29:46 2022
@author: ashrith

DATASET: Higgs-Boson from kaggel
Information on dataset:
training.csv - Training set of 250000 events, with an ID column, 30 feature columns, a weight column and a label column.
test.csv - Test set of 550000 events with an ID column and 30 feature columns
"""

import pandas as pd
import xgboost as xgb

# main only execs if this is the source file

if __name__ == '__main__':
    tests_df = \
        pd.read_csv('/home/ashrith/github/Higgs_XGboost/dataset/test.csv'
                    )
    train_df = \
        pd.read_csv('/home/ashrith/github/Higgs_XGboost/dataset/training.csv'
                    )

# Setting 'drop_first=True' we don't get 2 different columns for Label
# Rather one just for s (1 if s and 0 if b)

train_df = pd.get_dummies(train_df, columns=['Label'], drop_first=True)

ncols_train = train_df.shape[1]
ncols_tests = tests_df.shape[1]
Y_train = train_df.iloc[:, ncols_train - 1]
X_train = train_df.iloc[:, 1:ncols_train - 2]
Y_tests = tests_df.iloc[:, ncols_tests - 1]
X_tests = tests_df.iloc[:, 1:ncols_tests - 2]

weight = train_df.iloc[:, ncols_train - 2] * len(Y_tests) / len(Y_train)
sum_wpos = sum(weight[i] for i in range(len(Y_train)) if Y_train[i]
               == 1)  # b and s
sum_wneg = sum(weight[i] for i in range(len(Y_train)) if Y_train[i]
               == 0)

# Param list

param = {}
param['objective'] = 'binary:logistic'
param['scale_pos_weight'] = sum_wneg / sum_wpos
param['eta'] = 0.15
param['eval_metric'] = 'auc'
param['silent'] = 1
param['nthread'] = 1

# To watch muliple params use the following, else only displays last
# eval_metric
param_list = list(param.items()) + [('eval_metric', 'ams@')]
num_round = 360

# param['lambda']=0.1 (part of denominator in formula)
# param['gamma']=0.1 (if gain > gamma, then keep leaf)
# param['subsample'] = 0.5
# param['min_child_weight']= 0 (sum of cover should be less than this for leaf to exist)
# param['colsample_bytree']= 1 (0-1] range
# param['max_depth'] = 6
# param['early_stopping_rounds']=10
# param['seed'] = seed
# random.seed(seed)

# Model: Dmatrix

D_train = xgb.DMatrix(X_train, Y_train, missing=-999.0, weight=weight)
D_tests = xgb.DMatrix(X_tests, Y_tests)
watchlist_train = [(D_train, 'train')]
watchlist_tests = [(D_tests, 'tests')]

# this works

print('loading data end, start to boost trees')

boosted_tree = xgb.train(
    param_list,
    D_train,
    num_round,
    watchlist_train,
    early_stopping_rounds=20,
    verbose_eval=10,
)

boosted_tree.save_model('higgs.model')

print("finished training")
