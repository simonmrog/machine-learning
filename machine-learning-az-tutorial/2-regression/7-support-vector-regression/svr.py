#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#loading the dataset
dataset = pd.read_csv ("position-salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1:].values

#test-set
x_test = np.array ([[6.5]])

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler ()
sc_y = StandardScaler ()
x = sc_x.fit_transform (x)
x_test = sc_x.transform (x_test)
y = sc_y.fit_transform (y)

#fitting SVR to the dataset
from sklearn.svm import SVR
reg = SVR (kernel="rbf")
reg.fit (x, y)

#predicting a new result
y_pred = reg.predict (x_test)
y_pred = sc_y.inverse_transform (y_pred)
acc = 1 - abs((160000 - y_pred) / 160000)
print (acc)

#plotting the results
plt.scatter (x, y, color="red")
plt.plot (x, reg.predict (x), color="blue")
plt.show ()