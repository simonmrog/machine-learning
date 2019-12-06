#%%importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%loading the dataset
dataset = pd.read_csv ("co2emissions.csv")
x = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values
x_to_predict = np.array ([2.4, 4, 9.2]).reshape (1, -1)

# %%train-test-split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.15)

# %%fitting the model to data
from sklearn.linear_model import LinearRegression
model = LinearRegression ()
model.fit (x_train, y_train)
coeff = model.coef_
print (coeff)

# %%testing the model
y_pred = model.predict (x_test)
acc = model.score (x_test, y_test)
print ("{}%".format (acc * 100))

# %%
y_predicted = model.predict (x_to_predict)
print (y_predicted)
print (x_to_predict @ coeff)

