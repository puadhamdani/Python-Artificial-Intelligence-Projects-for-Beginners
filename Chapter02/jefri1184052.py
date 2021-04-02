import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def preparation():
    data = pd.read_csv('Chapter01/dataset/cancer.txt', sep=',', usecols=[0,1,2,3,4,5,6,7], header=None, names=['age', 'gender', 'persistentcough', 'Bleeding', 'alump', 'digestivedisorders','increasedheartrate', 'Diagnosis'])
   
    data = data.sample(frac=1)
    data = [data.iloc[:,:6], data.iloc[:, 6:]]


    df_dt_atribut = data.pop(0)
    df_dt_varr = data.pop(0)
    

    length = int(len(df_dt_varr)*0.75)

    df_trn_Varr = df_dt_varr[:length]
    df_trn_atribut = df_dt_atribut[:length]

    df_test_Varr = df_dt_varr[length:]
    df_test_atribut = df_dt_atribut[length:]

    return [[df_trn_atribut, df_trn_Varr], [df_test_atribut, df_test_Varr]]

def training(df_trn_atribut, df_trn_Varr):
    t = RandomForestClassifier(max_features=4, random_state=0, n_estimators=100)
    t = t.fit(df_trn_atribut, df_trn_Varr)
    return t

def testing(t, df_test_atribut):
    return t.predict(df_test_atribut)