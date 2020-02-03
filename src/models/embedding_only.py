import pandas as pd

from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
import pickle
import os

from src.core.plot import plot_graphs, plot_metrics
from src.core.metrics import F1Score

import src.constants.files as files
import src.constants.columns as c
import src.constants.models as m


def train_embedding_only_model():
    # TODO: docstring
    with open(files.NORMALIZED_TEST, "rb") as f:
        xtest_pad = pickle.load(f)

    with open(files.NORMALIZED_TRAIN, "rb") as f:
        xtrain_pad = pickle.load(f)

    with open(files.EMBEDDING_MATRIX, "rb") as f:
        embedding_matrix = pickle.load(f)

    targets = pd.read_csv(files.TRAIN_DATA, usecols=[c.Tweet.TARGET])

    # create a simple 3 layer sequential neural net
    model = Sequential()
    model.add(
        Embedding(embedding_matrix.shape()[0], m.EMBEDDING_DIM, weights=[embedding_matrix],
                  input_length=m.SENT_INPUT_LENGTH, trainable=False))
    # TODO: model inputs as variables
    model.add(Bidirectional(LSTM(200, dropout=0.5, recurrent_dropout=0.5)))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.add(Dropout(0.5))
    model.add(Activation('sigmoid'))

    model.summary()
    checkpoint_cb = ModelCheckpoint(os.path.join(files.MODELS_PATH, "my_keras_model.h5"), save_best_only=True)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', F1Score()])

    history = model.fit(xtrain_pad, targets, batch_size=512, epochs=100, validation_split=0.8, callbacks=[checkpoint_cb])

    predict_test = model.predict(xtest_pad)

    test_ids = pd.read_csv(files.TEST_DATA, usecols=[c.Tweet.ID])
    data_frame_predict = pd.DataFrame({c.Tweet.ID: test_ids[c.Tweet.ID], c.Tweet.TARGET: predict_test})

    data_frame_predict.to_csv(path_or_buf="submission_test_V1.csv", index=False)

    # TODO: put in a separate evaluation script
    plot_graphs(history, "accuracy")
    # plot_graphs(history, "loss")
    plot_metrics(history)
