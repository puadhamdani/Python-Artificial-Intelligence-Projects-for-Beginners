# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 07:15:16 2021

@author: HP
"""

import pandas as pd

def preparation(datasetpath):
    dada = pd.read_csv(datasetpath, sep=',')
    len(dada)
   
    
    dada = dada.sample(frac=1)
    
    dada_train = dada[:500]
    dada_test = dada[500:]
    
    dada_train_att = dada_train.drop(['id'], axis=1) #fitur
    dada_train_pass = dada_train['id'] #label
    
    dada_test_att = dada_test.drop(['id'], axis=1)
    dada_test_pass = dada_test['id']
    
    dada_att = dada.drop(['id'], axis=1)
    dada_pass = dada['id']
    
    return dada_train_att,dada_train_pass,dada_test_att,dada_test_pass,dada_att,dada_pass

def training(dada_train_att,dada_train_pass):
    # fit a decision tree
    from sklearn import tree
    tahu = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    tahu = tahu.fit(dada_train_att, dada_train_pass)
    return tahu

def testing(tahu,testdataframe):
    return tahu.predict(testdataframe)