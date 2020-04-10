def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("income.csv",header = 0,delimiter = ", ")
print(data[['education','education-num']])



labels = data[['income']]
features = data[['age','capital-gain','capital-loss','hours-per-week','sex','education-num']]
features['sex'] = features['sex'].apply(lambda x: 1 if x=='Female' else 0)

print(data['native-country'].value_counts())

features['country-int'] = data['native-country'].apply(lambda x: 0 if x=='United-States' else 1)
print(features.columns)

train_d, test_d, train_l, test_l = train_test_split(features,labels,random_state = 1, test_size=0.2)


forest = RandomForestClassifier(random_state = 1, n_estimators = 100)

forest.fit(train_d,train_l)
print(forest.feature_importances_)
print(forest.score(test_d,test_l))







