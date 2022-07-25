import numpy as np
import pandas as pd
from sklearn import linear_model, tree
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:/Users/11453/PycharmProjects/riskassessment/data/Treynor (Fund).csv")

# pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 20)

cor = df.corr()
cor_target = abs(cor["category_treynor_ratio_10years"])
# Selecting highly correlated features
relevant_features = cor_target[cor_target > 0.1]
# print(relevant_features.sort_values())

# print(df.isnull().sum().sort_values())

df = df.loc[:, ["asset_others", "price_earnings_ratio", 'sector_communication_services', "credit_b",
                "category_treynor_ratio_10years"]]

df = df.dropna()
X = df.iloc[:,0:3]
Y = df.iloc[:,4]

# split train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# logistic
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)
pred = model.predict(X_train)
rmse = mean_squared_error(pred, Y_train)**0.5
print(rmse)


# Ridge
model = linear_model.Ridge()
model.fit(X_train, Y_train)
pred = model.predict(X_train)
rmse = mean_squared_error(pred, Y_train)**0.5
print(rmse)

# Decision Tree
model2 = tree.DecisionTreeRegressor()
model2.fit(X_train, Y_train)
tree_pred = model2.predict(X_train)
tree_rmse = mean_squared_error(tree_pred,Y_train) ** 0.5
print(tree_rmse)
