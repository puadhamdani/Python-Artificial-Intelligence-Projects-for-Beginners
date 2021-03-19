from sklearn import tree #memanggil library tree dari sklrearn
import pandas as pd #memanggil library pandas

def prepoc(datapath):
    nanas = pd.read_csv(datapath, sep=',') #menampung file csv dengan variable nanas
    len(nanas) #mengjitung csv

    nanas = nanas.sample(frac=1) #memilih contoh
    nanas_train = nanas[:201] #mengambil data training
    nanas_test = nanas[200:] #mengambil data testing

    nanas_train_att = nanas_train.drop(['blue'], axis=1) #memanggil baris blue pada csv pada training
    nanas_train_pass = nanas_train['blue'] #memanggil baris blue pada csv trainig
    
    nanas_test_att = nanas_test.drop(['blue'], axis=1) #memanggil data label blue pada csv training
    nanas_test_pass = nanas_test['blue'] #memanggil data label blue pada csv training
    
    nanas_att = nanas.drop(['blue'], axis=1) #memanggil data label blue pada csv training
    nanas_pass = nanas['blue'] #memanggil data label blue pada csv training
    return nanas_train_att,nanas_train_pass,nanas_test_att,nanas_test_pass,nanas_att,nanas_pass #mengembalikan nilai variable


def training(nanas_train_att, nanas_train_pass): #mengatur data training
    apel = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5) #mengatur data yang ditampilkan dan mengatur maksimal ketika di tampilkan
    apel = apel.fit(nanas_train_att, nanas_train_pass)
    return apel

def testing(apel,testdataframe): #mengatur data testing
    return apel.predict(testdataframe)