import joblib
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("C:/Users/11453/PycharmProjects/riskassessment/data/Sharpe.csv")
df = df.dropna()
df.describe()
X = df.iloc[:, 3:6]
Y = df.iloc[:, 6]
# print(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)
pred = model.predict(X_train)
rmse = mean_squared_error(pred, Y_train) ** 0.5
print(rmse)

pred = model.predict(X_test)
rmse = mean_squared_error(pred, Y_test) ** 0.5
print(rmse)

# save model
joblib.dump(model, "sharpe ration")

# load model
model1 = joblib.load('sharpe ration')
pred1 = model1.predict([[0, 840000, 110]])
print(pred1)
