import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargando la base de datos de los pasajeros
passengers = pd.read_csv("passengers.csv")

#print(passengers.columns)
print(passengers.head())

######### Limpiando la base de datos
passengers['Sex'] = passengers['Sex'].apply(lambda x: 1 if x=='female' else 0)

# Eliminando NaN por la edad promedio
mean_age = passengers['Age'].mean(axis=0)
passengers['Age'].fillna(mean_age,inplace = True)
#print(passengers['Age'].values)

# Reorganizando datos...
passengers['FirstClass'] = passengers['Pclass'].apply(lambda x: 1 if x ==1 else 0)

# Second class column
passengers['SecondClass'] = passengers['Pclass'].apply(lambda x: 1 if x ==2 else 0)
print(passengers.head())

# Características importantes para hacer la regresión
features = passengers[['Sex','Age','FirstClass','SecondClass']]
survival = passengers['Survived']

# Separando datos de prueba y entrenamiento
f_train, f_test, s_train,s_test = train_test_split(features,survival,test_size=0.2,random_state = 100)
print(f_test.shape,f_train.shape)
# Reescalando los datos para no tener problemas de escala media = 0 y standard deviation = 1
scaler = StandardScaler()
f_train = scaler.fit_transform(f_train)
f_test = scaler.transform(f_test)

###### Regresión Logistica
model = LogisticRegression()
model.fit(f_train,s_train)

# Precisión de la data de entrenamiento
print(model.score(f_train,s_train))

# Precisión de la data de prueba
print(model.score(f_test,s_test))

# Coeficientes de la regresión.
### Coeficientes << 1 => mayor probabilidad clase positiva
### Coeficientes >> -1 => mayor probabilidad clase negativa|
print(model.coef_)

#### Haciendo predicciones
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
me = np.array([0.,5.,1.,0.])

# Combinando arreglos
sample_passengers = np.array([Jack,Rose,me])

# Reescalando
sample_passengers = scaler.transform(sample_passengers)

# Prediciendo !
print(model.predict(sample_passengers))
print(model.predict_proba(sample_passengers))

