import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

df = pd.read_csv('DBS.csv')
# print(df)
X = df.loc[:, "SGD"]
y = df.loc[:, "DBS"]
X = np.array(X).reshape(-1,1)

# print(Y)
model = linear_model.LogisticRegression()
model.fit(X, y.astype('string'))
pred = model.predict(X)
rmse = mean_squared_error(y, pred) ** 0.5
print(rmse)
