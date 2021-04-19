# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 09:15:10 2021

@author: DyningAida
"""
import random, os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def preparation():
    #word2vec
    import re
    data_list = list()
    i = 0
    for dirname in ["dataset/food"]:
        for fname in sorted(os.listdir("Chapter03/"+dirname)):
            if fname[-4:] == '.txt':
                with open("Chapter03/"+dirname+"/"+fname) as f:
                    for line in f:
                        sent = re.sub(r'[^\w\s]', '', line)
                        sent = sent.lower()
                        result = [w for w in sent.split(' ')]
                        data_list.append(result)
                        i += 1
    #doc2vec
    unsup_sentences = []
    for dirname in ["dataset/food"]:
        for fname in sorted(os.listdir("Chapter03/"+dirname)):
            if fname[-4:] == '.txt':
                with open("Chapter03/"+dirname+"/"+fname) as f:
                    sent = f.read()
                    words = sent.split()
                    unsup_sentences.append(TaggedDocument(words,[dirname+"/"+fname]))
    # shuffle data
    random.shuffle(unsup_sentences)
    
    return unsup_sentences, data_list

def training(data_list, unsup_sentences):
    from nltk.tokenize import word_tokenize
    from gensim.models import Word2Vec
    # buat model word2vec
    w2vmodel = Word2Vec(sentences=data_list, vector_size=52, workers=8)
    w2vmodel.wv.save_word2vec_format('w2v_model_1184030_review_amazon_food.bin', binary=True)
    # buat model doc2vec
    model = Doc2Vec(unsup_sentences,dm=0,hs=1,vector_size=20)
    model.build_vocab(unsup_sentences)
    model.save('d2v_model_1184030_review_food.d2v')
    return model, w2vmodel
    
def testing():
    from google_drive_downloader import GoogleDriveDownloader as gd
    import gensim, random
    from nltk.tokenize import word_tokenize
    from sklearn.ensemble import RandomForestClassifier
    #from sklearn.model_selection import train_test_split
    # download and load model
    gd.download_file_from_google_drive(file_id='1TG-z9iqgaj0QKnfMsXmITv1sB7U4V_Iv',
                                    dest_path='model/foodrev_model.d2v')
    w2vecmodel = gensim.models.KeyedVectors.load_word2vec_format('w2v_model_1184030_review_amazon_food.bin', binary=True)
    word = ['awesome','bad','good']
    for i in word:
        w2vecmodel[i]
        print(w2vecmodel[i])    
    # mengecek similaritas
    pembanding = ['bad','good']
    for i in pembanding:
        w2vec_test = print('similaritas awesome-',i,w2vecmodel.similarity('awesome',i))
    
    # download dan load doc2vec model
    gd.download_file_from_google_drive(file_id='1Jku7eBsA1zysfIrYGhWyKIiK3xckiJ1V',
                                    dest_path='model/d2v_model_1184030_review_food.d2v')
    model = gensim.models.doc2vec.Doc2Vec.load('model/d2v_model_1184030_review_food.d2v')
    # load dataset untuk prediksi
    gd.download_file_from_google_drive(file_id='1RPpTHPZYvTzg9OzYF4JAUWq5VTqowbH9',
                                    dest_path='model/dataset/food-reviews.csv')
    sentvecs = []
    sentences = []
    sentiments = []
    with open("model/dataset/food-reviews.csv") as f:
        for i, line in enumerate(f):
            line_split = line.strip().split(',')
            sentences.append(line_split[8])
            words = word_tokenize(line_split[8])
            sentvecs.append(model.infer_vector(words, steps=10)) #buat vektor dari dokumen ini
            sentiments.append(int(line_split[6]))
    # shuffle data
    random.shuffle(list(zip(sentences, sentvecs, sentiments)))
    sentences, sentvecs, sentiments = zip(*list(zip(sentences, sentvecs, sentiments)))
    # instansiasi random forest classifier
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(sentvecs, sentiments)
    # lakukan predict berdasar doc yang diinputkan
    vector = model.infer_vector(word_tokenize("i like this food so much"))
    # buat score
    score = clf.score(sentvecs, sentiments)
    return clf.predict([vector]), score


