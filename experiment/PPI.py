# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 19:49:51 2016

@author: zhouchang
"""

import pandas as pd
import numpy as np

#a=pd.read_csv('lag30.txt',sep='  ',names=np.arange(540))
#print a.shape[0]
#a['label']=1
#a[1458:]['label']=0
#a.to_csv('lag30.csv',sep=',',index=False)
#count=0
#summ=0
#for i in range(0,a['label'].shape[0]):
#    if a.iloc[i]['label']==1:
#        count=count+1
#    else:
#        summ=summ+1     
#print ('p:'),count,('n:'),summ
#print a
######################################################################
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import matthews_corrcoef 
from sklearn.metrics import  f1_score,precision_score,recall_score
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

t=pd.read_csv('yeast.csv',sep=',')
t=t.sample(frac=1)#random

alg=GradientBoostingClassifier(n_estimators=1200,max_depth=6,subsample=0.7,learning_rate=0.05)#parameter
#alg=RandomForestClassifier(n_estimators=1200,max_depth=30)                      
predictors=np.arange(694)

kf = KFold(t.shape[0], n_folds=5)
acc_mean=0
mcc_mean=0
f1_mean=0
sn_mean=0
ppv_mean=0
for train, test in kf:
    X_train, X_test = t[predictors].iloc[train], t[predictors].iloc[test]
    y_train, y_test = t['label'].iloc[train], t['label'].iloc[test]
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

    print("=========")
    

print 'Acc:',(acc_mean/5.0)
print 'Mcc:',(mcc_mean/5.0)
print 'F1:',(f1_mean/5.0)
print 'SN:',(sn_mean/5.0)
print 'PPV:',(ppv_mean/5.0)
                                           
########################################################################                                                    
#scores = cross_val_score(alg, t[predictors], t['label'],cv=5)
                                                   
                                                   
                                                    
                                                    
                                                    
                                                    
                                                    
                                                    
