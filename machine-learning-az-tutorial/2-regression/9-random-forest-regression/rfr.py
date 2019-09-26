#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading the dataset
dataset = pd.read_csv ("position-salaries.csv")
x_train = dataset.iloc[:, 1:-1].values
y_train = dataset.iloc[:, -1].values
x_test = np.array ([[6.5]])

#fitting the model to the data
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor (n_estimators=400, random_state=0)
model.fit (x_train, y_train)

#predicting results
y_pred = model.predict (x_test)
print (y_pred)

#plotting model and results
x_grid = np.arange (min (x_train), max (x_train), 0.01)
x_grid = x_grid.reshape (len (x_grid), 1)
plt.scatter (x_train, y_train, color="red")
plt.plot (x_grid, model.predict (x_grid), color="blue")
plt.show ()