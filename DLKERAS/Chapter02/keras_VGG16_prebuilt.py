from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np


# 사전에 구성되고 학습된 심층 학습 VGG16 모델
base_model = VGG16(weights='imagenet', include_top=True)
for i, layer in enumerate(base_model.layers):
	print (i, layer.name, layer.output_shape)

# block4_pool 블록에서 특징 추출
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

img_path = './data/cat.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 이 블록에서 특징을 추출
features = model.predict(x)

print(features)
