from keras.models import Model
from keras.preprocessing import image
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import numpy as np
import cv2

# imagenet에 대해 사전 학습된 가중치를 갖는 사전에 빌드된 모델
model = VGG16(weights='imagenet', include_top=True)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')

# VGG16 학습된 이미지 포맷으로 리사이즈
im = cv2.resize(cv2.imread('./data/steam-locomotive.jpg'), (224, 224))
im = np.expand_dims(im, axis=0)

# 예측
out = model.predict(im)
plt.plot(out.ravel())
plt.show()
print(np.argmax(out))
# 증기 기관차를 뜻하는 820이 나와야 한다.

