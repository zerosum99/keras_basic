
# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.layers import Merge
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Sequential

vocab_size = 5000
embed_size = 300

word_model = Sequential()
word_model.add(Embedding(vocab_size, embed_size, input_length=1, embeddings_initializer="glorot_uniform"))
word_model.add(Reshape((embed_size,)))

context_model = Sequential()
context_model.add(Embedding(vocab_size, embed_size, input_length=1, embeddings_initializer="glorot_uniform"))
context_model.add(Reshape((embed_size,)))

model = Sequential()
model.add(Merge([word_model, context_model], mode="dot", dot_axes=0))
model.add(Dense(1, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.compile(loss="mean_squared_error", optimizer="adam")

merge_layer = model.layers[0]
word_model = merge_layer.layers[0]
word_embed_layer = word_model.layers[0]
weights = word_embed_layer.get_weights()[0]
