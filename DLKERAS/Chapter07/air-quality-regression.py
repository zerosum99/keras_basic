# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

DATA_DIR = "../data"

AIRQUALITY_FILE = os.path.join(DATA_DIR, "AirQualityUCI.csv")

aqdf = pd.read_csv(AIRQUALITY_FILE, sep=";", decimal=",", header=0)
# 가장 앞 2열과 가장 뒤 2열 삭제
del aqdf["Date"]
del aqdf["Time"]
del aqdf["Unnamed: 15"]
del aqdf["Unnamed: 16"]
# 각 열의 NaN 값 들을 평균으로 채워 넣는다.
aqdf = aqdf.fillna(aqdf.mean())
# aqdf.head()

Xorig = aqdf.as_matrix()
Xorig.shape

scaler = StandardScaler()
Xscaled = scaler.fit_transform(Xorig)
# 처음 본 데이터를 예측할 때 사용하기 위해 저장한다.
Xmeans = scaler.mean_
Xstds = scaler.scale_

y = Xscaled[:, 3]
X = np.delete(Xscaled, 3, axis=1)

print(X.shape, y.shape, Xmeans.shape, Xstds.shape)

train_size = int(0.7 * X.shape[0])
Xtrain, Xtest, ytrain, ytest = X[0:train_size], X[train_size:], y[0:train_size], y[train_size:]
print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

# 네트워크 정의
readings = Input(shape=(12,))
x = Dense(8, activation="relu", kernel_initializer="glorot_uniform")(readings)
benzene = Dense(1, kernel_initializer="glorot_uniform")(x)

model = Model(inputs=[readings], outputs=[benzene])
model.compile(loss="mse", optimizer="adam")

# 네트워크 학습
NUM_EPOCHS = 20
BATCH_SIZE = 10

history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
                   validation_split=0.2)

# 일부 예측 값 시각화
ytest_ = model.predict(Xtest).flatten()
for i in range(10):
    label = (ytest[i] * Xstds[3]) + Xmeans[3]
    prediction = (ytest_[i] * Xstds[3]) + Xmeans[3]
    print("Benzene Conc. expected: {:.3f}, predicted: {:.3f}".format(label, prediction))
    
# 모든 예측 시각화
plt.plot(np.arange(ytest.shape[0]), (ytest * Xstds[3]) / Xmeans[3], 
         color="b", label="actual")
plt.plot(np.arange(ytest_.shape[0]), (ytest_ * Xstds[3]) / Xmeans[3], 
         color="r", alpha=0.5, label="predicted")
plt.xlabel("time")
plt.ylabel("C6H6 concentrations")
plt.legend(loc="best")
plt.show()
