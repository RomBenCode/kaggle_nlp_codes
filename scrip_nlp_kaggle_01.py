# script pour tester la competition kaggle sur un sujet nlp à savoir la detection de vrais desastres sur des tweets
# 14 janvier 2020
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from gensim.utils import tokenize
from gensim.models import word2vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from keras.utils import np_utils
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D, Input, concatenate
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer

stop_words = stopwords.words('english')

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

def plot_metrics(history):
    f1_score = history.history['f1_score']
    loss = history.history['loss']
    val_f1_score = history.history['val_f1_score']
    val_loss = history.history['val_loss']

    epochs = range(len(f1_score))

    plt.plot(epochs, f1_score, 'b', label='Training f1_score')
    plt.plot(epochs, val_f1_score, 'y', label='Validation f1_score')
    plt.title('F1 Score')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'y', label='Validation Loss')
    plt.title('loss')
    plt.legend()

    plt.show()

class F1Score(tf.keras.metrics.Metric):

    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.p = tf.keras.metrics.Precision()
        self.r = tf.keras.metrics.Recall()

    def update_state(self, *args, **kwargs):
        self.p.update_state(*args, **kwargs)
        self.r.update_state(*args, **kwargs)

    def reset_states(self):
        self.p.reset_states()
        self.r.reset_states()

    def result(self):
        p_res, r_res = self.p.result(), self.r.result()
        return (2 * p_res * r_res) / (p_res + r_res)
# this function creates a normalized vector for the whole sentence
def sent2vec(s):
    words = str(s).lower()#.decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
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

# Create a SentimentIntensityAnalyzer object.
sid_obj = SentimentIntensityAnalyzer()

flag_save_word2vec = False

train_valid_df = pd.read_csv("../nlp-getting-started/train.csv")
test_df = pd.read_csv("../nlp-getting-started/test.csv")

#----------------- Premiere tentative----------------------------
#----------------------------------------------------------------
sentance = [list(tokenize(s, deacc=True, lower=True)) for s in train_valid_df['text']]
print(sentance[1])

if flag_save_word2vec:
    model = word2vec.Word2Vec(sentance, size=300, window=20,
                          min_count=2, workers=1, iter=100)
    # save model
    model.save('model.bin')
else:
    # load model
    model = word2vec.Word2Vec.load('model.bin')

print(model.corpus_count)
print(model.wv['after'].shape, model.wv['after'][:10])

embeddings_index = {}
#print(model.wv.vocab)
for word in tqdm(model.wv.vocab):
    print(word)
    embeddings_index[word] = model[word]

# create sentence vectors using the above function for training and validation set
xtrain_word2vec = [sent2vec(x) for x in tqdm(train_valid_df["text"])]
xtest_word2vec = [sent2vec(x) for x in tqdm(test_df["text"])]

#print(xtrain_glove[0:10])

# scale the data before any neural net:
scl = preprocessing.StandardScaler()
xtrain_scl = scl.fit_transform(xtrain_word2vec)
xtest_scl = scl.fit_transform(xtest_word2vec)

# we need to binarize the labels for the neural net
ytrain_enc = np_utils.to_categorical(train_valid_df["target"])

# de
embedding_dim = 300
trunc_type ='post'
padding_type ='post'
oov_tok = "<OOV>"
tokenizer = Tokenizer(oov_token=oov_tok)
tokenizer.fit_on_texts(train_valid_df['text'])
tokenizer.fit_on_texts(test_df['text'])

word_index = tokenizer.word_index
vocab_size = len(word_index)
max_len = 70

xtrain_seq = tokenizer.texts_to_sequences(train_valid_df["text"])
xtest_seq = tokenizer.texts_to_sequences(test_df["text"])
# zero pad the sequences
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xtest_pad = sequence.pad_sequences(xtest_seq, maxlen=max_len)

# create an embedding matrix for the words we have in the dataset
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# create a simple 3 layer sequential neural net
model = Sequential()
model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=False))
model.add(Bidirectional(LSTM(200, dropout=0.5, recurrent_dropout=0.5)))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(2))
model.add(Dropout(0.5))
model.add(Activation('softmax'))

model.summary()
checkpoint_cb = ModelCheckpoint("my_keras_model.h5", save_best_only=True)

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', F1Score()])

history = model.fit(xtrain_pad, ytrain_enc, batch_size=512, epochs=100, validation_split=0.8, callbacks=[checkpoint_cb])

predict_test = model.predict(xtest_pad)

# on definit le vecteur on forcant la classe en fonction de la composante la plus elevee
predict_one_or_zero = []

for elem in predict_test:
    if elem[0] > elem[1]:
        predict_one_or_zero.append(0)
    else:
        predict_one_or_zero.append(1)

data_frame_predict = pd.DataFrame({"id": test_df["id"], "target": predict_one_or_zero})

data_frame_predict.to_csv(path_or_buf="submission_test_V1.csv", index=False)

plot_graphs(history, "accuracy")
#plot_graphs(history, "loss")
plot_metrics(history)
