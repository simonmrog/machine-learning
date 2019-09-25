#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#loading the dataset
dataset = pd.read_csv ("ecommerce-customers.csv")
x = dataset.iloc [:, 3:-1].values
y = dataset.iloc [:, -1:].values

#train-test splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2)

dataset.info ()
dataset.describe ()

sns.jointplot (dataset["Time on Website"], dataset["Yearly Amount Spent"])
sns.jointplot (dataset["Time on App"], dataset["Yearly Amount Spent"])
sns.jointplot (dataset["Time on App"], dataset["Length of Membership"])
sns.pairplot (dataset)

sns.lmplot (x="Length of Membership", y="Yearly Amount Spent", data=dataset)

#fitting linear regression to the data
from sklearn.linear_model import LinearRegression
model = LinearRegression ()
model.fit (x_train, y_train)
y_pred = model.predict (x_test)
acc = model.score (x_test, y_test)
print (acc)

#normality for the residual errors
sns.distplot ((y_test - y_pred))