# %%importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

# %%loading data
dataset = pd.read_csv ("../datasets/fuel-consumption.csv")
dataframe = dataset [["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_CITY", "FUELCONSUMPTION_HWY", "CO2EMISSIONS"]]

# %%train-test split
mask = np.random.rand (len (dataframe)) < 0.8
train_set = dataframe [mask].to_numpy ()
test_set = dataframe [~mask].to_numpy ()
x_train = train_set [:, :-1]
y_train = train_set [:, -1:]
x_test = test_set [:, :-1]
y_test = test_set [:, -1:]
# %%fitting model to data
from sklearn.linear_model import LinearRegression
model = LinearRegression ()
model.fit (x_train, y_train)
print ("Coefficients: ", model.coef_)


# %%prediction
y_hat = model.predict (x_test)
print ("R^2: %.2f" % model.score (x_test, y_test))
print ("MSE: %.2f" % np.mean ((y_hat - y_test) ** 2))

# %%
