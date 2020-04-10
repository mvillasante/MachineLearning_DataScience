import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv("passengers.csv")


# Update sex column to numerical
passengers['Sex'] = passengers['Sex'].apply(lambda x: 1 if x=='female' else 0)

# Fill the nan values in the age column
mean_age = passengers['Age'].mean(axis=0)
passengers['Age'].fillna(mean_age,inplace = True)
#print(passengers['Age'].values)

# Create a first class column
passengers['FirstClass'] = passengers['Pclass'].apply(lambda x: 1 if x ==1 else 0)

# Create a second class column
passengers['SecondClass'] = passengers['Pclass'].apply(lambda x: 1 if x ==2 else 0)
print(passengers)

# Select the desired features
features = passengers[['Sex','Age','FirstClass','SecondClass']]
survival = passengers['Survived']

# Perform train, test, split
f_train, f_test, s_train,s_test = train_test_split(features,survival,test_size=0.2,random_state = 100)
print(f_test.shape,f_train.shape)
# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
f_train = scaler.fit_transform(f_train)
f_test = scaler.transform(f_test)

# Create and train the model
model = LogisticRegression()
model.fit(f_train,s_train)

# Score the model on the train data
print(model.score(f_train,s_train))

# Score the model on the test data
print(model.score(f_test,s_test))

# Analyze the coefficients
print(model.coef_)

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
me = np.array([0.,5.,1.,0.])

# Combine passenger arrays
sample_passengers = np.array([Jack,Rose,me])

# Scale the sample passenger features
sample_passengers = scaler.transform(sample_passengers)

# Make survival predictions!
print(model.predict(sample_passengers))
print(model.predict_proba(sample_passengers))

