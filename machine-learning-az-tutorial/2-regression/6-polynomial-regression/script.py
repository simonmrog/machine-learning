#Comparison with a simple linear regression and a linear polynomial regression.
#The employee says it earns at 6.5 level, about 160.000 dollars
#According to the linear model, he earns 300.000 dollars and according to the polynomial model (which fits better to the data) he earns 158.000 dollars. He is saying the truth.

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv ("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression ()
lin_reg.fit (x, y)

#visualizing the linear regression results
plt.scatter (x, y, color="r")
plt.plot (x, lin_reg.predict (x))
plt.show ()
x_pred = np.array ([6.5]).reshape (1, 1)
y_pred1 = lin_reg.predict (x_pred)
print (y_pred1)

#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
pol_reg = PolynomialFeatures (degree=4)
x_poly = pol_reg.fit_transform (x)
model = LinearRegression ()
model.fit (x_poly, y)
print (model)   

#visualizing the polynomial regression results
x_grid = np.arange (min (x), max (x), 0.01)
x_grid = x_grid.reshape (len (x_grid), 1)
plt.scatter (x, y, color="r")
plt.plot (x_grid, model.predict (pol_reg.fit_transform (x_grid)), color="b")
plt.show ()
xpredpol = pol_reg.fit_transform ([[6.5]])
y_pred2 = model.predict (xpredpol)
print (y_pred2)