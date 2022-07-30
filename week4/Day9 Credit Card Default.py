import pandas as pd
from sklearn import cluster
from sklearn.metrics import confusion_matrix
from yellowbrick.cluster import KElbowVisualizer

df = pd.read_csv("C:/Users/11453/PycharmProjects/riskassessment/data/Credit Card Default II.csv")
df.dropna(inplace=True)

X = df.drop(columns=["clientid", "default"])
Y = df['default']

model = cluster.KMeans(n_clusters=2)
model.fit(X)
pred = model.predict(X)

# the end of KMeans
# check if keams accuracy is good
cm = confusion_matrix(Y, pred)
print((cm[0, 0] + cm[1, 1]) / sum(sum(cm)))


v = KElbowVisualizer(model, k=(2, 15))

v.fit(X)
v.show()
