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


train_root = "C:/Users/11453/PycharmProjects/riskassessment/data/Training/"
test_root = "C:/Users/11453/PycharmProjects/riskassessment/data/Test/"
# print(train_root)

# show a picture
# image = io.imread("C:/Users/11453/PycharmProjects/riskassessment/data/Training/rose/12240303_80d87f77a3_n.jpg")
# print(image.shape)
# print(image)
# io.imshow(image)
# plt.show() # show the picture

# In current set, the score is 0.687
batch_size = 2

Generator = ImageDataGenerator()
train_data = Generator.flow_from_directory(train_root, (100, 100), batch_size=batch_size)
test_data = Generator.flow_from_directory(test_root, (100, 100), batch_size=batch_size)


im = train_data[0][0][1]
img = tf.keras.preprocessing.image.array_to_img(im)
imshow(img)
# plt.show()

num_classes = len([i for i in os.listdir(train_root)])
# print(num_classes)

model = Sequential()

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
model.fit(train_data, batch_size = batch_size, epochs=10)

# score = model.evaluate(train_data)
# print(score)
score = model.evaluate(test_data)
print(score)

# save this model
# from keras.models import save_model
# save_model(model, "Flower")
