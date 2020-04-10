import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv("manhatan.csv")

print(df.head())

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = df[['rent']]

print(x['bedrooms'])
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2,random_state=6)

print(x_test['size_sqft']*2)
print(x_train.shape)	

mlr = LinearRegression()

mlr.fit(x_train,y_train)

y_predict = mlr.predict(x_test)

sonny_apartment = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]

predict = mlr.predict(sonny_apartment)

print("Predicted price: $%.2f" % predict)

#plt.scatter(y_test,y_predict,alpha=0.4)
#plt.xlabel("Actual prices")
#plt.ylabel("Predicted prices")
m = mlr.coef_[0,2]
b = mlr.intercept_
print(b)
#plt.scatter(df[['size_sqft']],df[['rent']],alpha = 0.4)
#plt.plot(x['size_sqft'],x['size_sqft']*m+b,color="red")



print(mlr.score(x_train,y_train))
print(mlr.score(x_test,y_test))

residuals = y_predict - y_test

plt.scatter(y_predict, residuals, alpha=0.4)
plt.title('Residual Analysis')

plt.show()

