import pandas
from sklearn import tree
from sklearn.model_selection import cross_val_score

def preparation():
    cvd = pandas.read_csv('Chapter01/dataset/datacovid.csv', sep=',')
    cvd = cvd.sample(frac=1)
    cvd_train = cvd[:126]
    cvd_test = cvd[127:]
    cvd_train_attribute = cvd_train.drop(['positive_covid'], axis=1)
    cvd_train_pos = cvd_train['positive_covid']
    cvd_test_attribute = cvd_test.drop(['positive_covid'], axis=1)
    cvd_test_pos = cvd_test['positive_covid']
    data = [[cvd_train_attribute,cvd_train_pos], [cvd_test_attribute, cvd_test_pos]]
    return data

def training(cvd_train_cd, cvd_train_pos):
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t = t.fit(cvd_train_cd,cvd_train_pos)
    return t

def testing(t, testdataframe):
    return t.predict(testdataframe)