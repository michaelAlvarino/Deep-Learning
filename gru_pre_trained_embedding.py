from os.path import join
import pandas as pd
import numpy as np
import gzip
import json
import time
import scipy.spatial
import pickle

from sklearn.feature_extraction import DictVectorizer
from keras.preprocessing.text import text_to_word_sequence

from time import time
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.merge import Dot
from keras.callbacks import EarlyStopping, TensorBoard
from keras import metrics
from keras import backend as K
from keras.models import Sequential
from keras.layers import GRU
from keras.layers.merge import Add, Dot, Concatenate
from keras.preprocessing.sequence import pad_sequences

# utils
def initEmbeddingMap(fileName):
    print("initializing embeddings")
    with open(join("data", "glove.6B", fileName)) as glove:
        return {l[0]: np.asarray(l[1:], dtype="float32") for l in [line.split() for line in glove]}

def clean(text):
    return text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                 lower=True, split=" ")


# returns dict with users and movies they rated as repeated rows
# cleans review text and add to rawOutput
def initRawData(input_file):
    print("initializing raw data")
    rawInputData = []
    rawOutputData = []
    with open(input_file,"r") as f:
        for i in f:
            line = f.readline()
            if len(line) < 4:
                break
            lineObj = json.loads(line)
            user = lineObj["reviewerID"]
            item = lineObj["asin"]
            rawInputDataObj = {"user": user, "asin": item}
            rawOutputDataObj = clean(lineObj["reviewText"])
            rawInputData.append(rawInputDataObj)
            rawOutputData.append(rawOutputDataObj)
    return rawInputData, rawOutputData

# creates dict of usrs w/ all movies rated + movies w/ all user ratings ***
def group_data(inputData):
    users = {}
    items = {}
    for datum in inputData:
        u = datum["user"]
        i = datum["asin"]
        users.setdefault(u, []).append(i)
        items.setdefault(i, []).append(u)
    return users, items

def getSetFromData(key, data):
    return set([datum.get(key) for datum in data])

def seq_2_matrix(sequence, embedding_map):
    return np.array([embedding_map.get(word) for word in sequence if word in embedding_map])

def matrix_2_avg(emb_matrix):
    return np.mean(emb_matrix, 0)


# utils - one hot encodes all data
def initVecData(rawInputData, rawOutputData, embedding_map):
    print('initializing vectorized data')
    dictVect = DictVectorizer()
    vecInputData = dictVect.fit_transform(rawInputData).toarray()
    vecOutputData = [matrix_2_avg(seq_2_matrix(review, embedding_map)) for review in rawOutputData]
    return vecInputData, vecOutputData

def initMatInputData(rawInputData, rawOutputData, embedding_map, save=False):
    print('initializing matrix data')
    if len(rawInputData) != len(rawOutputData):
        raise ValueError("Need same size of input and output")
    users = {}
    extra_info = {}
    items = {}
    dictVect = DictVectorizer()
    for i in range(len(rawInputData)):
        vecOutput = seq_2_matrix(rawOutputData[i], embedding_map)
        rawInput = rawInputData[i]
        user = rawInput['user']
        item = rawInput['asin']
        users.setdefault(user, []).append(vecOutput)
        items.setdefault(item, []).append(vecOutput)

    matUserInputData = []
    matItemInputData = []
    users = {k: np.vstack(v) for k, v in users.items()}
    items = {k: np.vstack(v) for k, v in items.items()}
    extra_info['user_seq_sizes'] = [m.shape[0] for m in users.values()]
    extra_info['item_seq_sizes'] = [m.shape[0] for m in items.values()]
    for i in range(len(rawInputData)):
        rawInput = rawInputData[i]
        user = rawInput['user']
        item = rawInput['asin']
        matUserInputData.append(users.get(user))
        matItemInputData.append(items.get(item))
    return matUserInputData, matItemInputData, extra_info

def toKey(user, item):
    return (user, item)

def initRatingsOutputData(rawInputData, input_file, save=False):
    ratingsData = []
    userItemDict = {}
    for i in range(len(rawInputData)):
        rawInput = rawInputData[i]
        userItem = toKey(rawInput['user'], rawInput['asin'])
        userItemDict[userItem] = i
        ratingsData.append(None) # check later to make sure no Nones left

    with open(input_file,'r') as f:
        for i in f:
            line = f.readline()
            lineObj = json.loads(line)
            user = lineObj['reviewerID']
            item = lineObj['asin']
            rating = lineObj['overall']
            i = userItemDict.get(toKey(user, item))
            if i is not None:
                ratingsData[i] = rating
        failure = None in ratingsData
        if failure:
            raise ValueError(str(len([r for r in ratingsData if r is None])) + " reviews did not have corresponding rating.")
    return ratingsData

fileName = "data/reviews_Amazon_Instant_Video_5.json"
rawInputData, rawOutputData = initRawData(input_file=fileName)

users, movies = group_data(rawInputData)

rand_idxs = np.random.permutation(len(rawOutputData))
rawInputData = [rawInputData[i] for i in rand_idxs]
rawOutputData = [rawOutputData[i] for i in rand_idxs]

embedding_map = initEmbeddingMap("glove.6B.100d.txt")

all_users = getSetFromData('user', rawInputData)
all_movies = getSetFromData('asin', rawInputData)
vecInputData, vecOutputData = initVecData(rawInputData, rawOutputData, embedding_map)

matUserInputData, matMovieInputData, extra_info = initMatInputData(rawInputData, rawOutputData, embedding_map)

matUserInputData[:10]

fileName = "data/reviews_Amazon_Instant_Video_5.json"
ratingsData = initRatingsOutputData(rawInputData, input_file=fileName,save=False)

class DeepCoNN():
    def __init__(self, embedding_size, hidden_size, rnn_hidden_size, u_seq_len, m_seq_len, filters=2, kernel_size=8,
                 strides=6):
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.filters = filters
        self.kernel_size = kernel_size
        self.inputU, self.towerU = self.create_deepconn_tower(u_seq_len)
        self.inputM, self.towerM = self.create_deepconn_tower(m_seq_len)
        self.joined = Concatenate()([self.towerU, self.towerM])
        self.outNeuron = Dense(1)(self.joined)

    def create_deepconn_tower(self, max_seq_len):
        input_layer = Input(shape=(max_seq_len, self.embedding_size))
        tower = GRU(self.rnn_hidden_size, dropout=0.1, activation="tanh")(input_layer)
        tower = Dense(self.hidden_size, activation="relu")(tower)
        return input_layer, tower

    def create_deepconn_dp(self):
        dotproduct = Dot(axes=1)([self.towerU, self.towerM])
        output = Add()([self.outNeuron, dotproduct])
        self.model = Model(inputs=[self.inputU, self.inputM], outputs=[output])
        self.model.compile(optimizer='Adam', loss='mse')

    def train(self, matUserInputData, matItemInputData, ratingsData, u_seq_len=200, i_seq_len=200, epochs=3500, training=None):
        tensorboard = TensorBoard(log_dir="gru_100dim_tf_logs/{}".format(time()))
        self.create_deepconn_dp()

        self.user_input = pad_sequences(np.asarray(matUserInputData), maxlen=u_seq_len)
        self.item_input = pad_sequences(np.asarray(matItemInputData), maxlen=i_seq_len)

        self.trainingN = int(len(user_input) * training) if type(training) is float else training

        self.outputs = np.asarray(ratingsData)
        print(self.model.summary())

        self.train_inputs = [self.user_input[:self.trainingN], self.item_input[:self.trainingN]]
        self.train_outputs = self.outputs[:self.trainingN]
        self.test_inputs = [self.user_input[self.trainingN:], self.item_input[self.trainingN:]]
        self.test_outputs = self.outputs[self.trainingN:]

        early_stopping = EarlyStopping(monitor='loss', patience=4)
        early_stopping_val = EarlyStopping(monitor='val_loss', patience=6)

        batch_size = 32

        self.history = self.model.fit(self.train_inputs, self.train_outputs, callbacks=[early_stopping, early_stopping_val, tensorboard], validation_split=0.2, batch_size=batch_size, epochs=epochs)
        self.predicts = self.model.predict(self.test_inputs)


# Calculates median user review length and item length. We then pad each review to these numbers
ptile = 50
u_seq_len = int(np.percentile(np.array(extra_info['user_seq_sizes']), ptile))
i_seq_len = int(np.percentile(np.array(extra_info['item_seq_sizes']), ptile))
embed_dims = matUserInputData[0].shape[1]
hidden_size = 64
rnn_hidden_size = 64
deepconn = DeepCoNN(embed_dims, hidden_size, rnn_hidden_size, u_seq_len, i_seq_len)

deepconn.train(matUserInputData, matMovieInputData, ratingsData,
           u_seq_len=u_seq_len, i_seq_len=i_seq_len,
           epochs=20, training=None)

deepconn.model.save("model_gru_100emb.h5")
