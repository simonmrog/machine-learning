# %%importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%loading the dataset and exploring data
dataset = pd.read_csv ("fuel-consumption.csv")
dataset.head ()

# %%summary of the data
dataset.describe ()


# %%selecting some features
dataframe = dataset [["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]]
dataframe.head (9)

dataframe.hist ()

# %%plotting fuel consumption vs co2 emissions
plt.scatter (dataframe.FUELCONSUMPTION_COMB, dataframe.CO2EMISSIONS, color="blue")
plt.xlabel ("Ful Consumption")
plt.ylabel ("Emissions")
plt.show ()

# %%
x = dataframe.ENGINESIZE.to_numpy (copy=True)
X = x.reshape (-1, 1)
y = dataframe.CO2EMISSIONS.to_numpy (copy=True)

# %%plotting features
plt.scatter (X, y)
plt.xlabel ("ENGINESIZE")
plt.ylabel ("CO2EMISSIONS")
plt.plot ()

# %%train-test-splitting
mask = np.random.rand (len (dataset)) < 0.8
x_train = X[mask]
x_test = X[~mask]
y_train = y[mask]
y_test = y[~mask]

# %%model data
from sklearn.linear_model import LinearRegression
model = LinearRegression ()
model.fit (x_train, y_train)
print ("Coefficients: ", model.coef_)
print ("Intercept: ", model.intercept_)
y_pred = model.predict (x_test)

# %%plotting results
plt.scatter (x, y, color="blue")
plt.plot (x_test, y_pred, color="red")
plt.xlabel ("ENGINESIZE")
plt.ylabel ("CO2EMISSIONS")
plt.show ()

# %%evaluating the model
from sklearn.metrics import r2_score

MAE = np.mean (np.absolute (y_pred - y_test))
MSE = np.mean ((y_pred - y_test) ** 2)
RMSE = np.sqrt (np.mean ((y_pred - y_test) ** 2))
R2 = model.score (x_test, y_test)

print ("MAE: ", MAE)
print ("MSE: ", MSE)
print ("RMSE: ", RMSE)
print ("R2: ", R2)


# %%
