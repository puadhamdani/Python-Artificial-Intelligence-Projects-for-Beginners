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
    
    def test_02_yusufjordan_1184026(self):
        from Chapter01.YusufJordan1184026 import preparation, training, testing
        datasetpath = 'Chapter01/dataset/test.csv'
        # testing function preparation
        dada_train_att, dada_train_pass, dada_test_att, dada_test_pass, dada_att, dada_pass = preparation(datasetpath)
        #testing function training
        tahu = training(dada_train_att, dada_train_pass)
        #testing function testing
        hasiltestingsemua = testing(tahu, dada_test_att)
        #hasil
        print('\n hasil testing yusuf : ')
        print(hasiltestingsemua)
        ambilsatuhasiltesting = hasiltestingsemua[0]
        self.assertLessEqual(ambilsatuhasiltesting, 1)
        


