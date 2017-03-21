# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 11:50:58 2016

@author: zhouchang
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import matthews_corrcoef 
from sklearn.metrics import  f1_score,precision_score,recall_score
from sklearn.cross_validation import cross_val_score

#a=pd.read_csv('AC.txt',sep='  ',names=np.arange(360))
#print a.shape[0]
#a['label']=1
#a[3899:]['label']=0
#a.to_csv('AC_human.csv',sep=',',index=False)
#count=0
#summ=0
#for i in range(0,a['label'].shape[0]):
#    if a.iloc[i]['label']==1:
#        count=count+1
#    else:
#        summ=summ+1     
#print ('p:'),count,('n:'),summ
#print a
#################################sample#################################################
t=pd.read_csv('train.csv',sep=',')
t=t.sample(frac=1)
ti=pd.read_csv('test.csv',sep=',')
ti=ti.sample(frac=1)

alg=GradientBoostingClassifier(n_estimators=2200,max_depth=6,subsample=0.7,learning_rate=0.05)#parameter
predictors=np.arange(694)
#predictions = []

acc_mean=0
mcc_mean=0
f1_mean=0
sn_mean=0
ppv_mean=0

X_train=t[predictors]
y_train=t['label']

X_test=ti[predictors]
y_test = ti['label']

print ('fit')
alg.fit(X_train, y_train)
print ('predict')
test_predictions = alg.predict(X_test.astype(float))[:]
accuracy = sum(test_predictions == y_test)/float(len(y_test))
print("Accuracy: {:.4f}".format(accuracy))
acc = alg.score(X_test, y_test)
print("Accuracy: {:.4f}".format(acc)) 
test_predictions[test_predictions==y_test]=1 
test_predictions[test_predictions!=y_test]=0
ti['result']=test_predictions                                
print ti['result']  
ti=ti['result']
ti.to_csv('false.csv',sep=',')   
  