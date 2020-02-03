import numpy as np
import pandas as pd
from gensim.utils import tokenize
from gensim.models import word2vec
from nltk import word_tokenize
from sklearn import preprocessing
from keras.preprocessing import sequence
from keras.utils import np_utils
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm
import pickle

import src.constants.files as files
import src.constants.columns as c
import src.constants.models as m


def preprocess():
    train_df = pd.read_csv(files.TRAIN_DATA)
    test_df = pd.read_csv(files.TEST_DATA)

    sentence = [list(tokenize(s, deacc=True, lower=True)) for s in train_df[c.Tweet.TEXT]]

    model = word2vec.Word2Vec(sentence, size=300, window=20, min_count=2, workers=1, iter=100)

    embeddings_index = {}
    for word in tqdm(model.wv.vocab):
        embeddings_index[word] = model[word]

    tokenizer = Tokenizer(oov_token=m.OOV_TOK)
    tokenizer.fit_on_texts(train_df[c.Tweet.TEXT])

    # create an embedding matrix for the words we have in the dataset
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, m.EMBEDDING_DIM))
    for word, i in tqdm(word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    xtrain_seq = tokenizer.texts_to_sequences(train_df[c.Tweet.TEXT])
    xtest_seq = tokenizer.texts_to_sequences(test_df[c.Tweet.TEXT])
    # zero pad the sequences
    xtrain_pad = sequence.pad_sequences(
        xtrain_seq, maxlen=m.SENT_INPUT_LENGTH, padding=m.PADDING_TYPE, truncating=m.TRUNC_TYPE)
    xtest_pad = sequence.pad_sequences(
        xtest_seq, maxlen=m.SENT_INPUT_LENGTH, padding=m.PADDING_TYPE, truncating=m.TRUNC_TYPE)

    with open(files.NORMALIZED_TRAIN, "wb") as f:
        pickle.dump(xtrain_pad, f)

    with open(files.NORMALIZED_TEST, "wb") as f:
        pickle.dump(xtest_pad, f)

    with open(files.EMBEDDING_MATRIX, "wb") as f:
        pickle.dump(embedding_matrix, f)


def sent2vec(s, embeddings_index):
    """
    Create a normalized vector for the whole sentence.

    :param s: sentence as string.
    :param embeddings_index: dictionnary {word: embedding of word} for all words in training data
    :return: normalized vector
    """
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in m.EN_STOP_WORDS]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())

