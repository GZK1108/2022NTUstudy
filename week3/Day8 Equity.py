from collections import Counter
import statsmodels.api as sm
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from imblearn.over_sampling import SMOTE
from sklearn import linear_model, ensemble, neural_network, tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

df = pd.read_csv("C:/Users/11453/PycharmProjects/riskassessment/data/Equity.csv")

print(df.isnull().any())  # judge data is null or not
d = df.isnull().sum().sort_values()  # sort
# print(d)
c = df.corr()
c = abs(c['Class'])  # correlation with Class

df = df.loc[:,
     ["Class", "Dividend per Share", "10Y Net Income Growth (per Share)", "10Y Shareholders Equity Growth (per Share)",
      "EBIT", "Net Income", "Sector"]]

df.dropna(inplace=True)  # Removing rows of NAN
df.boxplot()
# plt.show()

# DUMMY
dummy = pd.get_dummies(df["Sector"])
df = df.merge(dummy, left_index=True, right_index=True)
df = df.drop("Sector", axis="columns")

z = stats.zscore(df.astype(np.float))  # zscore conversion need float
z = np.abs(z)  # convert all to positive because the parity is not important
f = (z < 4).all(axis=1)  # 3 is your choice, axis =1 means by columns, f is a flag
df = df[f]

y = df.loc[:, ['Class']]
X = df.drop(columns='Class')

# normolization
X["Net Income"] = stats.zscore(X["Net Income"].astype(np.float))
X["EBIT"] = stats.zscore(X["EBIT"].astype(np.float))

# QQ plot -check heteroskedasticity-residual must be normally distributed
mod_fit = sm.OLS(y,X).fit()
res = mod_fit.resid
fig = sm.qqplot(res)
# plt.show()

# anova
model = ols("y ~ X", df).fit()
print(model.summary())
anova_results = anova_lm(model)
print('\nANOVA results')
print(anova_results)

# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# use smote to balance data
print('Original dataset shape %s' % Counter(y_train))
smo = SMOTE(random_state=42)  # 42 Control the randomization of the algorithm.
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

# save model
# joblib.dump(model, "randomforest_equity.j1")

# load model
# model1 = joblib.load('randomforest_quity.j1')
# pred1 = model1.predict(x_test)

# gradientboost
model = ensemble.GradientBoostingClassifier(max_depth=10)
model.fit(X_train, y_train)
pred = model.predict(X_test)
cm = confusion_matrix(y_test, pred)
print("gradientboost accuracy", (cm[0, 0] + cm[1, 1]) / (sum(sum(cm))))

# decision tree
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
cm = confusion_matrix(y_test, pred)
print("decisiontree accuracy", (cm[0, 0] + cm[1, 1]) / (sum(sum(cm))))


# MLP
model = neural_network.MLPClassifier(solver="lbfgs", hidden_layer_sizes=(100, 100), max_iter=500)
model.fit(X_train, y_train)
pred = model.predict(X_test)
cm = confusion_matrix(y_test, pred)
print("MLP accuracy", (cm[0, 0] + cm[1, 1]) / (sum(sum(cm))))
