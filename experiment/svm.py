# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 09:52:12 2016

@author: zhouchang
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.metrics import matthews_corrcoef 
from sklearn.metrics import  f1_score,precision_score,recall_score
from sklearn.grid_search import GridSearchCV

t=pd.read_csv('yeast.csv',sep=',')
t=t.sample(frac=1)#random


predictors=np.arange(694)

#predictions = []

param_grid= {'C':[9.0,10.0,11.0,12.0],
              'gamma': [0.35,0.45,0.55]}#parameter

gridcv= GridSearchCV(SVC(kernel='rbf'), param_grid)
alg=SVC(C=3.5,gamma=0.65,kernel='rbf')#matine
#alg=SVC(C=10.75,gamma=0.5,kernel='rbf')#yeast
kf = KFold(t.shape[0], n_folds=5)
acc_mean=0
mcc_mean=0
f1_mean=0
sn_mean=0
ppv_mean=0
for train, test in kf:
    X_train, X_test = t[predictors].iloc[train], t[predictors].iloc[test]
    y_train, y_test = t['label'].iloc[train], t['label'].iloc[test]
#    gridcv.fit(X_train, y_train)
#    print("Best params:"),(gridcv.best_params_)
#    print("Best score:"),(gridcv.best_score_)
#   print("Best estimator:"),(gridcv.best_estimator_)
#    print("~~~~~~~~~~~~~~~~~~~~~~")
#    c_best = gridcv.best_params_["C"]
#    g_best = gridcv.best_params_["gamma"]
#    print ("C:"),c_best,("g:"),g_best
#    alg=SVC(kernel='rbf', C=c_best, gamma=g_best)
    alg.fit(X_train, y_train)
    test_predictions = alg.predict(X_test.astype(float))[:]
    accuracy = sum(test_predictions == y_test)/float(len(y_test))
    print("Accuracy: {:.4f}".format(accuracy))
    
    acc = alg.score(X_test, y_test)
    print("Accuracy: {:.4f}".format(acc)) 
    acc_mean=acc_mean+acc
    
    mcc=matthews_corrcoef(y_test,test_predictions)
    print("MCC: {:.4f}".format(mcc))
    mcc_mean=mcc_mean+mcc
    
    f1=f1_score(y_test,test_predictions)
    print("F1: {:.4f}".format(f1))
    f1_mean=f1_mean+f1
    
    sn=recall_score(y_test,test_predictions)
    print("SN: {:.4f}".format(sn))
    sn_mean=sn_mean+sn
    
    ppv=precision_score(y_test,test_predictions)
    print("PPV: {:.4f}".format(ppv))
    ppv_mean=ppv_mean+ppv

    print("=======================")
    

print 'Acc:',(acc_mean/5.0)
print 'Mcc:',(mcc_mean/5.0)
print 'F1:',(f1_mean/5.0)
print 'SN:',(sn_mean/5.0)
print 'PPV:',(ppv_mean/5.0)