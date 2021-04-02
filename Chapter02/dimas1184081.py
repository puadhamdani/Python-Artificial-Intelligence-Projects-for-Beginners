# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:57:12 2021

@author: ASUS
"""

import pandas
from sklearn.ensemble import RandomForestClassifier

def preparation():
    d = pandas.read_csv('Chapter01/dataset/heart_failure_clinical_records_dataset.txt', sep='\s+', header=None, error_bad_lines=False, warn_bad_lines=False,
                     names=['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time', 'DEATH_EVENT'])
    d = d.sample(frac=1)
    
    d_attribute = d.iloc[:, :12]
    d_var = d.iloc[:, 12:]

    d_train_attribute = d_attribute[:148]
    d_train_var = d_var[:148]
    d_test_attribute = d_attribute[148:]
    d_test_var = d_var[148:]

    d_train_var = d_train_var['DEATH_EVENT']
    d_test_var = d_test_var['DEATH_EVENT']

    data = [[d_train_attribute,d_train_var], [d_test_attribute, d_test_var]]
    return data

def training(d_train_attribute, d_train_var):
    t = RandomForestClassifier(max_features=2, random_state=0, n_estimators=100)
    t = t.fit(d_train_attribute, d_train_var)
    return t

def testing(t, testdataframe):
    return t.predict(testdataframe)