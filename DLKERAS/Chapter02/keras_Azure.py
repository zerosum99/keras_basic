# 스크립트는 이 모듈의 엔트리 포인트인 azureml_main
# 이라는 함수를 반드시 포함해야 한다.

# 사용할 라이브러리 불러오기
import pandas as pd
import theano
import theano.tensor as T
from theano import function

from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
# 엔트리 포인트 함수는 최대 두 개의 입력 인자를 포함 할 수 있다.
#   Param<dataframe1>: pandas.DataFrame
#   Param<dataframe2>: pandas.DataFrame
def azureml_main(dataframe1 = None, dataframe2 = None):
    # 실행 로직은 여기서 시작
    # print('Input pandas.DataFrame #1:rnrn{0}'.format(dataframe1))
    # zip 파일이 세 번째 입력 포트에 연결된 경우,
    # ".Script Bundle"에 압축을 풀고 이 디렉토리를 sys.path 에 추가한다.
    # zip 파일에 파이썬 파일 mymodule.py이 포함되어 있는 경우
    # import mymodule를 통해 불러와서 사용할 수 있다.
    model = Sequential()
    model.add(Dense(1, input_dim=784, activation="relu"))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    data = np.random.random((1000,784))
    labels = np.random.randint(2, size=(1000,1))
    model.fit(data, labels, nb_epoch=10, batch_size=32)
    model.evaluate(data, labels)
    
    return dataframe1,
