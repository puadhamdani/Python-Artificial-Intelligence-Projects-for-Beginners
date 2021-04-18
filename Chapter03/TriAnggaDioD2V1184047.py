def w2v_preprocessing():
    import bz2
    import re

    review_data_list = list()

    i = 0

    with bz2.open('train.ft.txt.bz2', 'rt') as train_file:
        for line in train_file:
            if i == 1000000:
                break

            remove_punctuation = re.sub(r'[^\w\s]', '', line[11:])

            lower_all_characters = remove_punctuation.lower()

            result = [w for w in lower_all_characters.split(' ')]

            review_data_list.append(result)

            i += 1

    return review_data_list


def w2v_training(training_data_list):
    from gensim.models import Word2Vec

    dimension = 300

    model = Word2Vec(sentences=training_data_list, vector_size=dimension, workers=8)
    model.wv.save_word2vec_format('gensim_w2v_model_1184047_review_amazon.bin', binary=True)
    return model


def w2v_testing():
    import os

    try:
        os.mkdir('model')
    except:
        pass

    from google_drive_downloader import GoogleDriveDownloader

    GoogleDriveDownloader.download_file_from_google_drive(
        file_id='16lUqfemZkzufM8mGZ_Lu7BRfsObAc-4T',
        dest_path='model/model_w2v_1184047.bin'
    )

    from gensim.models import KeyedVectors

    data = {
        'match_word': {
            'word_1': 'cat',
            'word_2': 'dog'
        },
        'unmatch_word': {
            'word_1': 'freeze',
            'word_2': 'clock'
        }
    }

    model = KeyedVectors.load_word2vec_format('model/model_w2v_1184047.bin', binary=True)

    match_word = model.similarity(data['match_word']['word_1'], data['match_word']['word_2'])
    print(f"word 1: {data['match_word']['word_1']}, word 2: {data['match_word']['word_2']}, similarity: {match_word}")

    unmatch_word = model.similarity(data['unmatch_word']['word_1'], data['unmatch_word']['word_2'])
    print(f"word 1: {data['unmatch_word']['word_1']}, word 2: {data['unmatch_word']['word_2']}, similarity: {unmatch_word}")

    return match_word

def d2v_preprocessing():
    import gensim
    import bz2

    review_data_list = list()

    i = 0

    with bz2.open('train.ft.txt.bz2', 'rt') as train_file:
        for enum, line in enumerate(train_file):
            if i == 1000000:
                break

            normalize_words = gensim.utils.simple_preprocess(line[11:])

            review_data_list.append(gensim.models.doc2vec.TaggedDocument(normalize_words, [enum]))

            i += 1

    return review_data_list


def d2v_training(review_data_list):
    from gensim.models import Doc2Vec
    dimension = 52
    model = Doc2Vec(review_data_list, size=dimension, workers=8)
    model.save('model_w2v_1184047.model')
    return model


def d2v_testing():
    import os

    try:
        os.mkdir('model')
    except:
        pass

    from google_drive_downloader import GoogleDriveDownloader

    GoogleDriveDownloader.download_file_from_google_drive(
        file_id='1-C8r45fI2lkUdSHhMkqPjXOCWp2qHe1h',
        dest_path='model/model_d2v_1184047.d2v'
    )

    from gensim.models import Doc2Vec

    model = Doc2Vec.load('model/model_d2v_1184047.d2v')

    from sklearn.metrics.pairwise import cosine_similarity

    result_match = cosine_similarity(
        [model.infer_vector(['great', 'cd', 'my', 'lovely', 'pat', 'has', 'one', 'of', 'the', 'great', 'voices', 'of', 'her', 'generation', 'have', 'listened', 'to', 'this', 'cd', 'for', 'years', 'and', 'still', 'love', 'it', 'when', 'in', 'good', 'mood', 'it', 'makes', 'me', 'feel', 'better', 'bad', 'mood', 'just', 'evaporates', 'like', 'sugar', 'in', 'the', 'rain', 'this', 'cd', 'just', 'oozes', 'life', 'vocals', 'are', 'jusat', 'stuunning', 'and', 'lyrics', 'just', 'kill', 'one', 'of', 'life', 'hidden', 'gems', 'this', 'is', 'desert', 'isle', 'cd', 'in', 'my', 'book', 'why', 'she', 'never', 'made', 'it', 'big', 'is', 'just', 'beyond', 'me', 'everytime', 'play', 'this', 'no', 'matter', 'black', 'white', 'young', 'old', 'male', 'female', 'everybody', 'says', 'one', 'thing', 'who', 'was', 'that', 'singing'])],
        [model.infer_vector(['one', 'of', 'the', 'best', 'game', 'music', 'soundtracks', 'for', 'game', 'didn', 'really', 'play', 'despite', 'the', 'fact', 'that', 'have', 'only', 'played', 'small', 'portion', 'of', 'the', 'game', 'the', 'music', 'heard', 'plus', 'the', 'connection', 'to', 'chrono', 'trigger', 'which', 'was', 'great', 'as', 'well', 'led', 'me', 'to', 'purchase', 'the', 'soundtrack', 'and', 'it', 'remains', 'one', 'of', 'my', 'favorite', 'albums', 'there', 'is', 'an', 'incredible', 'mix', 'of', 'fun', 'epic', 'and', 'emotional', 'songs', 'those', 'sad', 'and', 'beautiful', 'tracks', 'especially', 'like', 'as', 'there', 'not', 'too', 'many', 'of', 'those', 'kinds', 'of', 'songs', 'in', 'my', 'other', 'video', 'game', 'soundtracks', 'must', 'admit', 'that', 'one', 'of', 'the', 'songs', 'life', 'distant', 'promise', 'has', 'brought', 'tears', 'to', 'my', 'eyes', 'on', 'many', 'occasions', 'my', 'one', 'complaint', 'about', 'this', 'soundtrack', 'is', 'that', 'they', 'use', 'guitar', 'fretting', 'effects', 'in', 'many', 'of', 'the', 'songs', 'which', 'find', 'distracting', 'but', 'even', 'if', 'those', 'weren', 'included', 'would', 'still', 'consider', 'the', 'collection', 'worth', 'it'])]
    )
    print(f'match: {result_match[0][0]}')

    result_unmatch = cosine_similarity(
        [model.infer_vector(['great', 'cd', 'my', 'lovely', 'pat', 'has', 'one', 'of', 'the', 'great', 'voices', 'of', 'her', 'generation', 'have', 'listened', 'to', 'this', 'cd', 'for', 'years', 'and', 'still', 'love', 'it', 'when', 'in', 'good', 'mood', 'it', 'makes', 'me', 'feel', 'better', 'bad', 'mood', 'just', 'evaporates', 'like', 'sugar', 'in', 'the', 'rain', 'this', 'cd', 'just', 'oozes', 'life', 'vocals', 'are', 'jusat', 'stuunning', 'and', 'lyrics', 'just', 'kill', 'one', 'of', 'life', 'hidden', 'gems', 'this', 'is', 'desert', 'isle', 'cd', 'in', 'my', 'book', 'why', 'she', 'never', 'made', 'it', 'big', 'is', 'just', 'beyond', 'me', 'everytime', 'play', 'this', 'no', 'matter', 'black', 'white', 'young', 'old', 'male', 'female', 'everybody', 'says', 'one', 'thing', 'who', 'was', 'that', 'singing'])],
        [model.infer_vector(['dvd', 'player', 'crapped', 'out', 'after', 'one', 'year', 'also', 'began', 'having', 'the', 'incorrect', 'disc', 'problems', 'that', 've', 'read', 'about', 'on', 'here', 'the', 'vcr', 'still', 'works', 'but', 'hte', 'dvd', 'side', 'is', 'useless', 'understand', 'that', 'dvd', 'players', 'sometimes', 'just', 'quit', 'on', 'you', 'but', 'after', 'not', 'even', 'one', 'year', 'to', 'me', 'that', 'sign', 'on', 'bad', 'quality', 'giving', 'up', 'jvc', 'after', 'this', 'as', 'well', 'sticking', 'to', 'sony', 'or', 'giving', 'another', 'brand', 'shot'])]
    )
    print(f'unmatch: {result_unmatch[0][0]}')

    return result_match[0][0]