import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix

train_root = "C:/Users/11453/PycharmProjects/riskassessment/data/Dataset_BUSI_with_split/train/"
test_root = "C:/Users/11453/PycharmProjects/riskassessment/data/Dataset_BUSI_with_split/test/"

# batch_size = 1, accuracy is still not the same
batch_size = 1

Generator = ImageDataGenerator()
train_data = Generator.flow_from_directory(train_root, (150, 150), batch_size=batch_size)
test_data = Generator.flow_from_directory(test_root, (150, 150), batch_size=batch_size)

num_classes = len([i for i in os.listdir(train_root)])
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.05))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.05))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation="softmax"))
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(train_data, batch_size=batch_size, epochs=1)

# score = model.evaluate(train_data)
# print(score)
score = model.evaluate(test_data)
print(score)

pred = model.predict(test_data)
pred_classes = np.argmax(pred, axis=1)
cm = confusion_matrix(test_data.classes, pred_classes)
# print(cm)
print((cm[0,0]+cm[1,1]+cm[2,2])/(sum(sum(cm))))
sns.heatmap(cm,annot=True)
plt.show()