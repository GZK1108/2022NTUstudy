import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("C:/Users/11453/PycharmProjects/riskassessment/data/Movie.csv")
df = df.replace("yes", 1)
df = df.replace("no", 0)

item = apriori(df, use_colnames=True, min_support=0.2)
rules = association_rules(item, metric='confidence', min_threshold=0.7)
print(rules)
