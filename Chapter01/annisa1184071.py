# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 20:37:00 2021

@author: ASUS
"""

from sklearn import tree
import pandas as pd

def prepoc(datapath):
    jambi = pd.read_csv(datapath, sep=',')
    len(jambi)
    
    # shuffle data
    jambi = jambi.sample(frac=1)
    jambi_train = jambi[:101]
    jambi_test = jambi[50:]

    jambi_train_att = jambi_train.drop(['Units'], axis=1)
    jambi_train_pass = jambi_train['Units']
    
    jambi_test_att = jambi_test.drop(['Units'], axis=1)
    jambi_test_pass = jambi_test['Units']
    
    jambi_att = jambi.drop(['Units'], axis=1)
    jambi_pass = jambi['Units']
    return jambi_train_att,jambi_train_pass,jambi_test_att,jambi_test_pass,jambi_att,jambi_pass

def training(jambi_train_att,jambi_train_pass):
    batanghari = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    batanghari = batanghari.fit(jambi_train_att, jambi_train_pass)
    return batanghari

def testing(batanghari,testdataframe):
    return batanghari.predict(testdataframe)