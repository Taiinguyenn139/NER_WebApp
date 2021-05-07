import pandas as pd
import numpy as np
import os
import pickle
import nltk
import matplotlib.pyplot as plt
from nltk import word_tokenize
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import LSTM, Dense, TimeDistributed, Embedding, Bidirectional, Input
from keras.models import Model, load_model #Input
from keras_contrib.layers import CRF
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import f1_score
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from keras.preprocessing.text import text_to_word_sequence


def load_data(filename='ner_dataset.csv'):
    df = pd.read_csv(filename, encoding = "ISO-8859-1")
    df = df.fillna(method = 'ffill')
    return df

df = load_data()

def pad(n):
    must = len(n) - 75
    if(must < 0):
        for i in range(must*(-1)):
            n.append(0)
        return n
    elif(must == 0):
        return n
    else:
        return n[:75]


def build_model(num_tags, hidden_size = 50):
    # Model architecture
    input = Input(shape=(75,))
    model = Embedding(input_dim=35178 + 2, output_dim=40, input_length=75, mask_zero=False)(input)
    model = Bidirectional(LSTM(units=hidden_size, return_sequences=True, recurrent_dropout=0.1))(model)
    model = TimeDistributed(Dense(num_tags+1, activation="softmax"))(model)
    crf = CRF(num_tags + 1)  # CRF layer
    out = crf(model)  # output

    model = Model(input, out)
    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

    return model

def predict(df,sentence, model):
    text_word = word_tokenize(sentence)
    words = list(df['Word'].unique())
    tags = list(df['Tag'].unique())

    for i in range(len(text_word)):
        if text_word[i] not in words:
            text_word[i] = "UNK"

    word2idx = {w : i + 2 for i, w in enumerate(words)}
    word2idx["UNK"] = 1
    word2idx["PAD"] = 0
    tag2idx = {t : i + 1 for i, t in enumerate(tags)}
    tag2idx["PAD"] = 0
    idx2word = {i: w for w, i in word2idx.items()}
    idx2tag = {i: w for w, i in tag2idx.items()}
    X = [word2idx[w] for w in text_word]
    # Padding các câu về max_len
    X = pad(X)
    X = np.array(X)
    p = model.predict(X.reshape((1, 75)))
    p = np.argmax(p, axis=-1)
    aloha = [idx2tag[x] for x in p[0]]
    aloha = aloha[:len(text_word)]
    data = {'named entity':aloha, 'output':text_word}
    df = pd.DataFrame(data)
    return df

