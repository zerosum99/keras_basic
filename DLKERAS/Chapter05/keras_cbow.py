# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.core import Dense, Lambda
from keras.layers.embeddings import Embedding
import keras.backend as K

vocab_size = 5000
embed_size = 300
window_size = 1

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=300, input_length=2, embeddings_initializer="glorot_uniform"))

model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
model.add(Dense(5000, activation="softmax", kernel_initializer="glorot_uniform"))

model.compile(loss='categorical_crossentropy', optimizer="adadelta")

# 가중치 가져오기
weights = model.layers[0].get_weights()[0]