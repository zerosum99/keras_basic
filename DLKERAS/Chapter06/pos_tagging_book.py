# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.layers.core import Activation, Dense, RepeatVector, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import collections
import nltk
import numpy as np
import os


def parse_sentences(filename):
    word_freqs = collections.Counter()
    num_recs, maxlen = 0, 0
    fin = open(filename, "r")
    for line in fin:
        words = line.strip().lower().split()
        for word in words:
            word_freqs[word] += 1
        if len(words) > maxlen:
            maxlen = len(words)
        num_recs += 1
    fin.close()
    return word_freqs, maxlen, num_recs


def build_tensor(filename, numrecs, word2index, maxlen,
                 make_categorical=False, num_classes=0):
    data = np.empty((numrecs,), dtype=list)
    fin = open(filename, "rb")
    i = 0
    for line in fin:
        wids = []
        for word in line.strip().lower().split():
            if word in word2index:
                wids.append(word2index[word])
            else:
                wids.append(word2index["UNK"])
        if make_categorical:
            data[i] = np_utils.to_categorical(
                wids, num_classes=num_classes)
        else:
            data[i] = wids
        i += 1
    fin.close()
    pdata = sequence.pad_sequences(data, maxlen=maxlen)
    return pdata

########################## 메인 ##########################

DATA_DIR = "../data"

MAX_SEQLEN = 250
S_MAX_FEATURES = 5000
T_MAX_FEATURES = 45

EMBED_SIZE = 128
HIDDEN_SIZE = 64

BATCH_SIZE = 32
NUM_EPOCHS = 1

# NLTK 데이터셋에서 추출
if not os.path.exists(os.path.join(DATA_DIR, "treebank_sents.txt")):
    fedata = open(os.path.join(DATA_DIR, "treebank_sents.txt"), "w")
    ffdata = open(os.path.join(DATA_DIR, "treebank_poss.txt"), "w")
    sents = nltk.corpus.treebank.tagged_sents()

    for sent in sents:
        words, poss = [], []
        for word, pos in sent:
            if pos == "-NONE-":
                continue
            words.append(word)
            poss.append(pos)
        fedata.write("{:s}\n".format(" ".join(words)))
        ffdata.write("{:s}\n".format(" ".join(poss)))

    fedata.close()
    ffdata.close()

# 데이터 탐색, 상수 설정
s_wordfreqs, s_maxlen, s_numrecs = parse_sentences(
    os.path.join(DATA_DIR, "treebank_sents.txt"))
t_wordfreqs, t_maxlen, t_numrecs = parse_sentences(
    os.path.join(DATA_DIR, "treebank_poss.txt"))
print(len(s_wordfreqs), s_maxlen, s_numrecs,
      len(t_wordfreqs), t_maxlen, t_numrecs)
# num_recs: 3914
# SOURCE num_unique_words: 10947, max_words_per_sent: 249
# TARGET num_unique_words: 45, max_words_per_sent: 249

# 룩업 테이블
s_vocabsize = min(len(s_wordfreqs), S_MAX_FEATURES) + 2
s_word2index = {x[0]: i + 2 for i, x in
                enumerate(s_wordfreqs.most_common(S_MAX_FEATURES))}
s_word2index["PAD"] = 0
s_word2index["UNK"] = 1
s_index2word = {v: k for k, v in s_word2index.items()}

t_vocabsize = len(t_wordfreqs) + 1
t_word2index = {x[0]: i for i, x in
                enumerate(t_wordfreqs.most_common(T_MAX_FEATURES))}
t_word2index["PAD"] = 0
t_word2index["UNK"] = 1
t_index2word = {v: k for k, v in t_word2index.items()}

# 입력 생성
X = build_tensor(os.path.join(DATA_DIR, "treebank_sents.txt"),
                 s_numrecs, s_word2index, MAX_SEQLEN)
Y = build_tensor(os.path.join(DATA_DIR, "treebank_poss.txt"),
                 t_numrecs, t_word2index, MAX_SEQLEN,
                 True, t_vocabsize)

# 학습/테스트 셋 분할
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2,
                                                random_state=42)
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)

# 네트워크 정의
model = Sequential()
model.add(Embedding(s_vocabsize, EMBED_SIZE,
                    input_length=MAX_SEQLEN))
model.add(SpatialDropout1D(0.2))
# model.add(LSTM(HIDDEN_SIZE, dropout_W=0.2, dropout_U=0.2))
# model.add(GRU(HIDDEN_SIZE, dropout_W=0.2, dropout_U=0.2))
#model.add(Bidirectional(LSTM(HIDDEN_SIZE, dropout_W=0.2, dropout_U=0.2)))
model.add(Bidirectional(LSTM(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2)))


model.add(RepeatVector(MAX_SEQLEN))
# model.add(LSTM(HIDDEN_SIZE, return_sequences=True))
# model.add(GRU(HIDDEN_SIZE, return_sequences=True))
model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True)))
model.add(TimeDistributed(Dense(t_vocabsize)))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])

model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
          validation_data=[Xtest, Ytest])

score, acc = model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE)
print("Test score: %.3f, accuracy: %.3f" % (score, acc))