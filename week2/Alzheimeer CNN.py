from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

train_root = "C:/Users/11453/PycharmProjects/riskassessment/data/alzheimeer/train/"
test_root = "C:/Users/11453/PycharmProjects/riskassessment/data/alzheimeer/test/"

batch_size = 32

Generator = ImageDataGenerator()
train_data = Generator.flow_from_directory(train_root, (150, 150), batch_size=batch_size)
test_data = Generator.flow_from_directory(test_root, (150, 150), batch_size=batch_size)

num_classes = len([i for i in os.listdir(train_root)])


from keras.applications.resnet import ResNet50
import keras
restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(150,150,3))
restnet.summary()

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers
model1 = Sequential()
model1.add(restnet)
model1.add(Flatten())
model1.add(Dense(64, activation='relu'))
model1.add(Dropout(0.05))
model1.add(Dense(num_classes, activation='softmax'))
model1.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model1.fit(train_data, batch_size = batch_size, epochs=10)

score = model1.evaluate(test_data)
print(score)
