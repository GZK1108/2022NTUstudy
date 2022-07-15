from matplotlib import pyplot as plt
from skimage import io
from keras.preprocessing.image import ImageDataGenerator

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from matplotlib.pyplot import imshow
import os


train_root = "C:/Users/11453/PycharmProjects/riskassessment/data/flower/Training/"
test_root = "C:/Users/11453/PycharmProjects/riskassessment/data/flower/Test/"
# print(train_root)

# show a picture
# image = io.imread("C:/Users/11453/PycharmProjects/riskassessment/data/Training/rose/12240303_80d87f77a3_n.jpg")
# print(image.shape)
# print(image)
# io.imshow(image)
# plt.show() # show the picture

# In current set, the score is 0.687
batch_size = 32

Generator = ImageDataGenerator()
train_data = Generator.flow_from_directory(train_root, (100, 100), batch_size=batch_size)
test_data = Generator.flow_from_directory(test_root, (100, 100), batch_size=batch_size)


im = train_data[0][0][1]
img = tf.keras.preprocessing.image.array_to_img(im)
imshow(img)
# plt.show()

num_classes = len([i for i in os.listdir(train_root)])
# print(num_classes)

"""model = Sequential()

model.add(Conv2D(16, (3, 3), input_shape=(100, 100, 3), activation='relu')) # you can change (5,5) to (3,3)
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.05))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.05))

model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.05))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.05))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.05))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.05))

model.add(Dense(num_classes, activation="softmax"))
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(train_data, batch_size = batch_size, epochs=1)

# score = model.evaluate(train_data)
# print(score)
score = model.evaluate(test_data)
print(score)"""

# save this model
# from keras.models import save_model
# save_model(model, "Flower")

# Using Resnet
from tensorflow import keras
from keras.applications.resnet import ResNet50
#from keras.applications.resnet50 import ResNet50
from keras.models import Model
import keras
restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(100,100,3))
restnet.summary()

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
model1 = Sequential()
model1.add(restnet)
model1.add(Flatten())
model1.add(Dense(512, activation='relu'))
model1.add(Dropout(0.3))
model1.add(Dense(num_classes, activation='softmax'))
model1.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model1.fit(train_data, batch_size = batch_size, epochs=1)

score = model1.evaluate(test_data)
print(score)

