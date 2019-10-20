from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K


# 사전 학습된 베이스 모델 생성
base_model = InceptionV3(weights='imagenet', include_top=False)

# layer.name, layer.input_shape, layer.output_shape
('mixed10', [(None, 8, 8, 320), (None, 8, 8, 768), (None, 8, 8, 768),
(None, 8, 8, 192)], (None, 8, 8, 2048))
('avg_pool', (None, 8, 8, 2048), (None, 1, 1, 2048))
('flatten', (None, 1, 1, 2048), (None, 2048))
('predictions', (None, 2048), (None, 1000))

# 전역 지역 평균 풀링 계층을 추가
x = base_model.output
x = GlobalAveragePooling2D()(x) #  첫 계층으로 완전 연결 계층 추가
x = Dense(1024, activation='relu')(x) # 마지막에 200 클래스를 갖는 로지스틱 계층
predictions = Dense(200, activation='softmax')(x) # 학습할 모델
model = Model(inputs=base_model.input, outputs=predictions)

# 모든 합성곱 InceptionV3 계층을 고정
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일 (계층을 트레이닝하지 않도록 설정한 *후에* 실행)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 에폭에 걸쳐 새로운 데이터로 모델 학습 몇 번의  model.fit_generator(...)


# 모델 컴파일 (학습되지 않도록 계층을 고정한 후에 진행)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# 새로운 데이터에 대해 몇번의 에폭 동안 학습 model.fit_generator(...)

# 최상위 2개 인셉션 블록을 학습하기로 결정
# 앞 172개 계층을 고정하고 나머지를 해제
for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

# 낮은 학습율(learning rate)을 갖는 SGD 사용
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),loss='categorical_crossentropy')

# 모델을 다시 학습(이번에는 최상위 덴스 계층옆에 있는 2개 블록에 대해 미세조정(fine-tuning)
#model.fit_generator(...)
