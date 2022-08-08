from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix
import pandas as pd

svm = OneClassSVM(kernel='rbf', gamma=0.05, nu=0.45)

df = pd.read_csv("C:/Users/11453/PycharmProjects/riskassessment/data/Credit Card Default II.csv")
df = df.dropna()

x = df.loc[:, ["income", "age", "loan"]]
y = df.loc[:, ["default"]]

svm.fit(x)
pred = svm.predict(x)

pred = pd.DataFrame(pred)
pred = pred.replace(1, 0)  # 1 is normal and –1 is abnormal and is default
pred = pred.replace(-1, 1)


cm = confusion_matrix(y, pred)
accuracy = (cm[0, 0] + cm[1, 1]) / (sum(sum(cm)))
