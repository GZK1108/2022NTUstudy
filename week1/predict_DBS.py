import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import tree
df = pd.read_csv('DBS.csv')
X = df.loc[:, ["SGD"]]
y = df.loc[:, ["DBS"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# LinearRegression
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
# print(model.coef_)
linear_pred = model.predict(X_test)
linear_rmse = mean_squared_error(y_test, linear_pred) ** 0.5
print("Linear_rmse:", linear_rmse)

# Decision tree
model2 = tree.DecisionTreeRegressor()
model2.fit(X_train, y_train)
tree_pred = model2.predict(X_test)
tree_rmse = mean_squared_error(y_test, tree_pred) ** 0.5
print("Tree_rmse:", tree_rmse)