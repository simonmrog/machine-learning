#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading the dataset
dataset = pd.read_csv ("position-salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1:].values
x_test = np.array ([[6.5]])

#fitting the decision tree regressor to the data
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor (random_state=0)
regressor.fit (x, y)

#predicting results
y_pred = regressor.predict (x_test)

#plotting results
x_grid = np.arange (min (x), max (x), 0.01)
x_grid = x_grid.reshape (len (x_grid), 1)
plt.scatter (x, y, color="red")
plt.plot (x_grid, regressor.predict (x_grid), color="blue")
plt.show ()