#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
filename = "Salary_Data.csv"
df = pd.read_csv (filename)
X = df.iloc [:, :-1].values
Y = df.iloc [:, 1].values

#train-test splitting
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size=1/3, random_state=0)

#fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regression = LinearRegression ()
regression.fit (X_train, Y_train)

#predicting the test set results
Y_pred = regression.predict (X_test)

plt.plot (X_train, regression.predict (X_train))
plt.scatter (X_test, Y_test, color="red")
plt.title ("Salary vs Experience (Training set)")
plt.xlabel ("Years of Experience")
plt.ylabel ("Salary")
plt.show ()
