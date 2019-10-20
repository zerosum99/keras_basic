import numpy as np
import scipy.misc
from keras.models import model_from_json
from keras.optimizers import SGD

# 모델 불러오기
model_architecture = 'cifar10_architecture.json'
model_weights = 'cifar10_weights.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

# 이미지 불러오기
img_names = ['./data/cat2.jpg', './data/dog.jpg']
imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), (32, 32)),
                     (1, 0, 2)).astype('float32')
        for img_name in img_names]
imgs = np.array(imgs) / 255

# 학습
optim = SGD()
model.compile(loss='categorical_crossentropy', optimizer=optim,
              metrics=['accuracy'])

# 예측
predictions = model.predict_classes(imgs)
print(predictions)
