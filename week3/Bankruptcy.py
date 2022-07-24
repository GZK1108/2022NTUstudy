# DAY 7
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import neural_network
from keras.models import Sequential
from keras.layers import Dense, Dropout

df = pd.read_csv("C:/Users/11453/PycharmProjects/riskassessment/data/bankruptcy.csv")

X = df.drop(columns='class')
y = df.iloc[:, -1]

# select K best features
features = SelectKBest(score_func=f_regression, k=5)
fit = features.fit(X, y)
# print(df.columns, fit.scores_)
mask = fit.get_support()
# print(mask)
new_features = X.columns[mask]
# print(new_features[:])

# select new_festures
X = df.loc[:, ['Attr3', 'Attr16', 'Attr26', 'Attr35', 'Attr51']]
y.nunique()  # find out category

# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# use smote to balance data
print('Original dataset shape %s' % Counter(y_train))
smo = SMOTE(sampling_strategy='auto', random_state=42)  # 42 Control the randomization of the algorithm.
X_train, y_train = smo.fit_resample(X_train, y_train)
print('Resampled dataset shape %s' % Counter(y_train))

# logistic
model = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)
pred = model.predict(X_test)
cm = confusion_matrix(y_test, pred)
print("logistic accuracy", (cm[0, 0] + cm[1, 1]) / (sum(sum(cm))))

# randomforest
model = ensemble.RandomForestClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
cm = confusion_matrix(y_test, pred)
print("randomforest accuracy", (cm[0, 0] + cm[1, 1]) / (sum(sum(cm))))

# gradientboost
model = ensemble.GradientBoostingClassifier(max_depth=10)
model.fit(X_train, y_train)
pred = model.predict(X_test)
cm = confusion_matrix(y_test, pred)
print("gradientboost accuracy", (cm[0, 0] + cm[1, 1]) / (sum(sum(cm))))

# MLP
model = neural_network.MLPClassifier(solver="lbfgs", hidden_layer_sizes=(100, 100), max_iter=500)
model.fit(X_train, y_train)
pred = model.predict(X_test)
cm = confusion_matrix(y_test, pred)
print("MLP accuracy", (cm[0, 0] + cm[1, 1]) / (sum(sum(cm))))

# seqential
model = Sequential()
model.add(Dense(10, input_dim=5, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, input_dim=5, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, input_dim=5, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=10, epochs=10, verbose=1)
pred = model.predict(X_test)
pred_classes = np.argmax(pred, axis=1)
cm = confusion_matrix(X_test.classes, pred_classes)
# print(cm)
print((cm[0, 0] + cm[1, 1] + cm[2, 2]) / (sum(sum(cm))))
