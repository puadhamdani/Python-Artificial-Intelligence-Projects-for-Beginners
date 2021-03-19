from sklearn import tree
import pandas as pd #mengimport library pada Python yaitu pandas

def prepoc(datapath):
    pusing = pd.read_csv(datapath, sep=',') #pusing digunakan untuk pembacaan file csv
    len(pusing)

    # shuffle data
    pusing = pusing.sample(frac=1)
    pusing_train = pusing[:2000] #data training yang digunakan 1 sampai 2000 data
    pusing_test = pusing[2000:] #data testing yang digunakan mulai dari data ke 2000 sampai data terakhir

    pusing_train_att = pusing_train.drop(['UsingIP'], axis=1) #menggunakan label UsingIP karena merupakan salah satu atribut data pada datasets phishing
    pusing_train_pass = pusing_train['UsingIP']

    pusing_test_att = pusing_test.drop(['UsingIP'], axis=1)
    pusing_test_pass = pusing_test['UsingIP']

    pusing_att = pusing.drop(['UsingIP'], axis=1) 
    pusing_pass = pusing['UsingIP']
    return pusing_train_att,pusing_train_pass,pusing_test_att,pusing_test_pass,pusing_att,pusing_pass

def training(pusing_train_att,pusing_train_pass): #struktur data training
    kepala = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5) #kepala mendefinisikan training
    kepala = kepala.fit(pusing_train_att, pusing_train_pass)
    return kepala

def testing(kepala,testdataframe): #struktur data testing
    return kepala.predict(testdataframe) #test data frame bertujuan untuk generalisasi data dari matriks dengan kolom yang berbeda