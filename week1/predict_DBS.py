import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import tree
import joblib

df = pd.read_csv('DBS.csv')
X = df.loc[:, ["SGD"]]
y = df.loc[:, ["DBS"]]

# LinearRegression
model = linear_model.LinearRegression()
model.fit(X, y)
# print(model.coef_)
linear_pred = model.predict(X)
linear_rmse = mean_squared_error(y, linear_pred) ** 0.5
print("Linear_rmse:", linear_rmse)
joblib.dump(model, "regression.j1")

# Decision tree
model2 = tree.DecisionTreeRegressor()
model2.fit(X, y)
tree_pred = model2.predict(X)
tree_rmse = mean_squared_error(y, tree_pred) ** 0.5
print("Tree_rmse:", tree_rmse)
joblib.dump(model2, "decisiontree.j1")
