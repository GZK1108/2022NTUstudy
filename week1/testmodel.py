import joblib
model = joblib.load("regression.j1")
print(model.coef_)
print(model.predict([[1.4]]))
model2 = joblib.load("decisiontree.j1")
print(model2.predict([[1.4]]))