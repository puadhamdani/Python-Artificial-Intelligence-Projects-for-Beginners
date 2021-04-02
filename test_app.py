import unittest


class TestApp(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_02_rolly_113040087(self):
        from Chapter01.rolly113040087 import preparation,training,testing
        dataset='Chapter01/dataset/student-por.csv'
        d_train_att,d_train_pass,d_test_att,d_test_pass,d_att,d_pass= preparation(dataset)
        t = training(d_train_att,d_train_pass)
        hasiltestingsemua = 	testing(t,d_test_att)
        print('\n hasil testing : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)

    def test_02_alvian_1184077(self):
        from Chapter01.alvian1184077 import prepoc, training, testing
        datapath = 'Chapter01/dataset/csv_result-messidor_features.csv'
        d_train_att, d_train_pass, d_test_att, d_test_pass, d_att, d_pass = prepoc(datapath)
        t = training(d_train_att, d_train_pass)
        hasiltestingsemua = testing(t, d_test_att)
        print('\n Hasil testing : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 2)

    def test_02_mauliddhia_1184101(self):
        from Chapter01.mauliddhia1184101 import prepocesing, training, testing
        datapath = 'Chapter01/dataset/transformed_test.csv'
        d_train_att, d_train_pass, d_test_att, d_test_pass, d_att, d_pass = prepocesing(datapath)
        t = training(d_train_att, d_train_pass)
        hasiltestingsemua = testing(t, d_test_att)
        print('\n Hasil testing : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 2)

    def test_02_aditya_1184090(self):
        from Chapter01.Aditya1184090 import preparation,training,testing
        #data
        data = preparation()
        #train data
        train = data.pop(0)
        dfs_train_att = train.pop(0)
        dfs_train_win = train.pop(0)
        #test data
        test = data.pop(0)
        dfs_test_att = test.pop(0)
        dfs_test_win = test.pop(0)
        #training
        t = training(dfs_train_att, dfs_train_win)
        #predict
        result = testing(t,dfs_test_att)
        print("result : ")
        print(result)
        self.assertLessEqual(result[0], 2)

    def test_02_rizal_1184033(self):
        from Chapter01.rizalramadhan1184033 import prepoc, training, testing
        datapath = 'Chapter01/dataset/train.csv'
        nanas_train_att, nanas_train_pass, nanas_test_att, nanas_test_pass, nanas_att, nanas_pass = prepoc(datapath)
        apel = training(nanas_train_att, nanas_train_pass) 
        hasiltestingsemua = testing(apel, nanas_test_att) 
        print('\n Hasil testing : ') 
        print(hasiltestingsemua) 
        ambilsatuhasiltesting = hasiltestingsemua[1] 
        self.assertLessEqual(ambilsatuhasiltesting, 1)
        
    def test_02_dimas_1184081(self):
        from Chapter01.dimas1184081 import prepoc,training,testing
        dataset='Chapter01/dataset/Pokedex_Condensed_Numeric_Dataset.csv'
        anak_train_att,anak_train_pass,anak_test_att,anak_test_pass,anak_att,anak_pass= prepoc(dataset)
        it = training(anak_train_att,anak_train_pass)
        hasiltestingsemua = 	testing(it,anak_test_att)
        print('\n hasil testing : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 8)

    def test_02_yuki_1184053(self):
        from Chapter01.yuki1184053 import prepoc,training,testing
        dataset='Chapter01/dataset/bankruptcy_prediction.csv'
        pencak_train_att,pencak_train_pass,pencak_test_att,pencak_test_pass,pencak_att,pencak_pass= prepoc(dataset)
        silat = training(pencak_train_att,pencak_train_pass)
        hasiltestingsemua =     testing(silat,pencak_test_att)
        print('\n hasil testing : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)

    def test_02_nurul_1184038(self):
        from Chapter01.nurul1184038 import prepoc, training, testing
        datapath = 'Chapter01/dataset/student-mat-pass-or-fail.csv'
        d_train_att, d_train_pass, d_test_att, d_test_pass, d_att, d_pass = prepoc(datapath)
        t = training(d_train_att, d_train_pass)
        hasiltestingsemua = testing(t, d_test_att)
        print('\n Hasil testing : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)
        
    def test_03_aditya_1184090(self):
        from Chapter02.Aditya1184090 import preparation,training,testing
        #data
        data = preparation()
        #train data
        train = data.pop(0)
        d_train_attribute = train.pop(0)
        d_train_var = train.pop(0)
        #test data
        test = data.pop(0)
        d_test_attribute = test.pop(0)
        d_test_var = test.pop(0)
        #training
        t = training(d_train_attribute, d_train_var)
        #predict
        result = testing(t,d_test_attribute)
        print("result : ")
        print(result)
        print('Score:', t.score(d_test_attribute, d_test_var))
        self.assertLessEqual(result[0], 2)
        
    def test_03_yusuf_1184026(self):
        from Chapter02.Yusuf1184026 import preparation, training, testing
        dataset = 'Chapter01/dataset/car.txt'
        # testing function preparation
        df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label = preparation(dataset)
        # testing function training
        clf = training(df_train_att, df_train_label)
        # testing function testing
        hasil = testing(clf, df_test_att.head())
        # hasil testing
        print('\nhasil testing Yusuf :', hasil)
        print('Score:', clf.score(df_test_att, df_test_label))
    
    def test_03_rizalramadhan_1184033(self):
        from Chapter02.rizalramadhan1184033 import prepoc, training, testing
        datapath = 'Chapter01/dataset/mobile.txt'
        nanas_train_att, nanas_train_pass, nanas_test_att, nanas_test_pass, nanas_att, nanas_pass = prepoc(datapath)
        apel = training(nanas_train_att, nanas_train_pass)
        hasiltestingsemua = testing(apel, nanas_test_att)
        print('\n Hasil testing : ')
        print(hasiltestingsemua)
        print('Score:', apel.score(nanas_test_att, nanas_test_pass))
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 2)
        
    def test_03_kevin_1184049(self):
        from Chapter02.kevin1184049 import preparation, training, testing
        datapath = 'Chapter01/dataset/phishing.txt'
        d_train_att, d_train_pass, d_test_att, d_test_pass, d_att, d_pass = preparation(datapath)
        t = training(d_train_att, d_train_pass)
        hasiltestingsemua = testing(t, d_test_att)
        print('\n Hasil testing : ')
        print(hasiltestingsemua)
        print('Score:', t.score(d_test_att, d_test_pass))
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 2)
        
    def test_03_dian_1184095(self):
        from Chapter02.dian1184095 import prepoc, training, testing
        datapath = 'Chapter01/dataset/new_test.txt'
        d_train_att, d_train_pass, d_test_att, d_test_pass, d_att, d_pass = prepoc(datapath)
        t = training(d_train_att, d_train_pass)
        hasiltestingsemua = testing(t, d_test_att)
        print('\n Hasil testing : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[1]
        self.assertLessEqual(ambilsatuhasiltesting, 2)
        print('Score:', t.score(d_test_att, d_test_pass))
        
    def test_03_YukiArdiansyah_1184053(self):
        from Chapter02.YukiArdiansyah1184053 import preparation,training,testing
        #data
        data = preparation()
        #train data
        train = data.pop(0)
        yuki_train_attribute = train.pop(0)
        yuki_train_var = train.pop(0)
        #test data
        test = data.pop(0)
        yuki_test_attribute = test.pop(0)
        yuki_test_var = test.pop(0)
        #training
        ardi = training(yuki_train_attribute, yuki_train_var)
        #predict
        result = testing(ardi,yuki_test_attribute)
        print("Hasil : ")
        print(result)
        print('Score:', ardi.score(yuki_test_attribute, yuki_test_var))
        self.assertLessEqual(result[0], 2)
        
    def test_03_almi_1184043(self):
        from Chapter02.almi1184043 import preparation,training,testing
        data = preparation()
        train = data.pop(0)
        d_train_attribute = train.pop(0)
        d_train_var = train.pop(0)
        test = data.pop(0)
        d_test_attribute = test.pop(0)
        d_test_var = test.pop(0)
        t = training(d_train_attribute, d_train_var)
        result = testing(t,d_test_attribute.head())
        print("result : ")
        print(result)
        print('Score:', t.score(d_test_attribute, d_test_var))
        self.assertLessEqual(result[1], 7)
        
    def test_03_helmiazhar_1184013(self):
        from Chapter02.helmiazhar1184013 import preparation,training,testing
        #data
        data = preparation()
        #train data
        train = data.pop(0)
        d_train_attribute = train.pop(0)
        d_train_var = train.pop(0)
        #test data
        test = data.pop(0)
        d_test_attribute = test.pop(0)
        d_test_var = test.pop(0)
        #training
        t = training(d_train_attribute, d_train_var)
        #predict
        result = testing(t,d_test_attribute.head())
        print("result : ")
        print(result)
        print('Score:', t.score(d_test_attribute, d_test_var))
        self.assertLessEqual(result[0], 1)
        
    def test_03_nurulkamila_1184038(self):
        from Chapter02.nurul1184038 import preparation,training,testing
        #data
        data = preparation()
        #train data
        train = data.pop(0)
        d_train_attribute = train.pop(0)
        d_train_var = train.pop(0)
        #test data
        test = data.pop(0)
        d_test_attribute = test.pop(0)
        d_test_var = test.pop(0)
        #training
        t = training(d_train_attribute, d_train_var)
        #predict
        result = testing(t,d_test_attribute.head())
        print("result : ")
        print(result)
        print('Score:', t.score(d_test_attribute, d_test_var))
        self.assertLessEqual(result[0], 1)
        
    def test_03_murnialestari_1184006(self):
        from Chapter02.murnialestari1184006 import preparation, training, testing

        datasetpath = 'Chapter01/dataset/bill_authentication.txt'
        # testing function preparation
        df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label = preparation(datasetpath)
        # testing function training
        clf = training(df_train_att, df_train_label)
        # testing function testing
        hasiltesting = testing(clf, df_test_att.head())
        # hasil
        print('\nhasil testing murnia : ')
        print(hasiltesting)
        ambilsatuhasiltesting = hasiltesting[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)                                    
        print('Score:', clf.score(df_test_att, df_test_label))
        
    def test_03_dindaanik_1184003(self):
        from Chapter02.dindaanik1184003 import preparation, training, testing

        datasetpath = 'Chapter01/dataset/College_admission.txt'
        # testing function preparation
        df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label = preparation(datasetpath)
        # testing function training
        clf = training(df_train_att, df_train_label)
        # testing function testing
        hasiltesting = testing(clf, df_test_att.head())
        # hasil
        print('\nhasil testing dinda : ')
        print(hasiltesting)
        print('Score:', clf.score(df_test_att, df_test_label))
