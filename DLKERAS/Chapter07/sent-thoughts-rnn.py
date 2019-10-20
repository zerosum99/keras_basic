# -*- coding: utf-8 -*-
from __future__ import division, print_function
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.layers.core import RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing import sequence
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import codecs

def lookup_word2id(word):
    try:
        return word2id[word]
    except KeyError:
        return word2id["UNK"]

def load_glove_vectors(glove_file, word2id, embed_size):
    embedding = np.zeros((len(word2id), embed_size))
    fglove = codecs.open(glove_file, "r", encoding='latin-1')
    for line in fglove:
        cols = line.strip().split()
        word = cols[0]
        if embed_size == 0:
            embed_size = len(cols) - 1
        #if word2id.has_key(word):
        if word in word2id:
            vec = np.array([float(v) for v in cols[1:]])
        embedding[lookup_word2id(word)] = vec
    embedding[word2id["PAD"]] = np.zeros((embed_size))
    embedding[word2id["UNK"]] = np.random.uniform(-1, 1, embed_size)
    return embedding
    
def sentence_generator(X, embeddings, batch_size):
    while True:
        # loop once per epoch
        num_recs = X.shape[0]
        indices = np.random.permutation(np.arange(num_recs))
        num_batches = num_recs // batch_size
        for bid in range(num_batches):
            sids = indices[bid * batch_size : (bid + 1) * batch_size]
            Xbatch = embeddings[X[sids, :]]
            yield Xbatch, Xbatch

def compute_cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x, 2) * np.linalg.norm(y, 2))


############################### 메인  ###############################

DATA_DIR = "../data"

# 문장을 파싱하고 사전을 만든다.
word_freqs = collections.Counter()
ftext = open(os.path.join(DATA_DIR, "text.tsv"), "r")
sents = []
for line in ftext:
    docid, text = line.strip().split("\t")
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            word = word.lower()
            word_freqs[word] += 1
        sents.append(sent)
ftext.close()

VOCAB_SIZE = 5000
EMBED_SIZE = 100
LATENT_SIZE = 512
SEQUENCE_LEN = 50

BATCH_SIZE = 64
NUM_EPOCHS = 10

# word2id = collections.defaultdict(lambda: 1)
word2id = {}
word2id["PAD"] = 0
word2id["UNK"] = 1
for v, (k, _) in enumerate(word_freqs.most_common(VOCAB_SIZE - 2)):
    word2id[k] = v + 2
id2word = {v:k for k, v in word2id.items()}

print("vocabulary sizes:", len(word2id), len(id2word))

sent_wids = [[lookup_word2id(w) for w in s.split()]
                                   for s in sents]
sent_wids = sequence.pad_sequences(sent_wids, SEQUENCE_LEN)

# glove vectors를 불러와서 가중치 행렬에 넣는다.
embeddings = load_glove_vectors(os.path.join(
    DATA_DIR, "glove.6B.{:d}d.txt".format(EMBED_SIZE)), word2id, EMBED_SIZE)
print(embeddings.shape)

# 문장을 학습 데이터와 테스트 데이터로 나눈다.
train_size = 0.7
Xtrain, Xtest = train_test_split(sent_wids, train_size=train_size)
print("number of sentences: ", len(sent_wids))
print(Xtrain.shape, Xtest.shape)

# 학습 데이터와 테스트 데이터 제네레이터 정의
train_gen = sentence_generator(Xtrain, embeddings, BATCH_SIZE)
test_gen = sentence_generator(Xtest, embeddings, BATCH_SIZE)

# 오토인코더 네트워크 정의
inputs = Input(shape=(SEQUENCE_LEN, EMBED_SIZE), name="input")
encoded = Bidirectional(LSTM(LATENT_SIZE), merge_mode="sum", 
                        name="encoder_lstm")(inputs)
decoded = RepeatVector(SEQUENCE_LEN, name="repeater")(encoded)
decoded = Bidirectional(LSTM(EMBED_SIZE, return_sequences=True), 
                        merge_mode="sum",
                        name="decoder_lstm")(decoded)

autoencoder = Model(inputs, decoded)

autoencoder.compile(optimizer="sgd", loss="mse")

# 학습
num_train_steps = len(Xtrain) // BATCH_SIZE
num_test_steps = len(Xtest) // BATCH_SIZE
checkpoint = ModelCheckpoint(filepath=os.path.join(DATA_DIR, "sent-thoughts-autoencoder.h5"),
                            save_best_only=True)
history = autoencoder.fit_generator(train_gen, 
                                   steps_per_epoch=num_train_steps,
                                   epochs=NUM_EPOCHS,
                                   validation_data=test_gen,
                                   validation_steps=num_test_steps,
                                   callbacks=[checkpoint])

# 결과 시각화
plt.plot(history.history["loss"], color="g", label="train")
plt.plot(history.history["val_loss"], color="b", label="validation")
plt.ylabel("loss (MSE)")
plt.xlabel("epochs")
plt.legend(loc="best")
plt.show()

# 테스트 데이터에 대한 오토인코더 예측
test_inputs, test_labels = test_gen.__next__()
preds = autoencoder.predict(test_inputs)

# 오토인코더의 인코더 부분 추출
encoder = Model(autoencoder.input, autoencoder.get_layer("encoder_lstm").output)
#encoder.summary()    

# 원본과 오토인코딩된 벡터 사이의 차이 계산
k = 500
cosims = np.zeros((k))
i = 0
for bid in range(num_test_steps):
    xtest, ytest = test_gen.__next__()
    ytest_ = autoencoder.predict(xtest)
    Xvec = encoder.predict(xtest)
    Yvec = encoder.predict(ytest_)
    for rid in range(Xvec.shape[0]):
        if i >= k:
            break
        cosims[i] = compute_cosine_similarity(Xvec[rid], Yvec[rid])
        if i <= 10:
            print(cosims[i])
        i += 1
    if i >= k:
        break

plt.hist(cosims, bins=10, normed=True)
plt.xlabel("cosine similarity")
plt.ylabel("frequency")
