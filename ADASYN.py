from collections import Counter
import pandas as pd
from imblearn.over_sampling import ADASYN

df = pd.read_csv('./随机森林填充/temp3.csv')
header = df.head(0)

X = df.iloc[:, 1:]  # X为解释变量集
y = df.iloc[:, 0]  # y为结果集
print('Original dataset shape %s' % Counter(y))

ada = ADASYN(sampling_strategy='auto', random_state=42)  # 42 Control the randomization of the algorithm.
X_res, y_res = ada.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))

X_res.insert(0, 'TARGET', y_res, allow_duplicates=False)

X_res.to_csv("./随机森林填充/baladsvm3.csv", index=False)
