#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv ("startups.csv")
print (dataset.head ())
x = dataset.iloc [:, :-1].values
y = dataset.iloc [:, -1].values

#categorical features encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder ()
x[:, 3] = labelencoder.fit_transform (x[:, 3])
onehotencoder = OneHotEncoder (categorical_features=[3])
x = onehotencoder.fit_transform (x).toarray ()
x.shape

#avoiding the dummy variable trap
x = x[:, 1:]

#train-test splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=0)

#fitting multiple linear regression model to the training set
from sklearn.linear_model import LinearRegression
model = LinearRegression ()
model.fit (x_train, y_train)

#predicting the test results
y_pred = model.predict (x_test)
acc = model.score (x_test, y_test)
print (acc)

#building the optimal model using Backward Elimination
import statsmodels.regression.linear_model as sm
x = np.append (arr=np.ones ((x.shape[0], 1)).astype (int), values=x, axis=1)
x_opt = [0, 1, 2, 3, 4, 5]
model_opt = sm.OLS (endog=y, exog=x[:, x_opt]).fit ()
print (model_opt.summary ())
x_opt = [0, 1, 3, 4, 5]
model_opt = sm.OLS (endog=y, exog=x[:, x_opt]).fit ()
print (model_opt.summary ())
x_opt = [0, 3, 4, 5]
model_opt = sm.OLS (endog=y, exog=x[:, x_opt]).fit ()
print (model_opt.summary ())
x_opt = [0, 3, 5]
model_opt = sm.OLS (endog=y, exog=x[:, x_opt]).fit ()
print (model_opt.summary ())
x_opt = [0, 3]
model_opt = sm.OLS (endog=y, exog=x[:, x_opt]).fit ()
print (model_opt.summary ())
x_opt = [3]  #removing the dummy ones column
x = x[:, x_opt]

#train-test splitting
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler ()
x_train = scaler.fit_transform (x_train)
x_test = scaler.transform (x_test)

#optimized multiple linear regression model
model.fit (x_train, y_train)
y_pred_opt = model.predict (x_test)
acc_opt = model.score (x_test, y_test)
print (acc_opt)

print ((acc_opt - acc)/acc * 100, "%")

#plotting results
plt.scatter (x_train, y_train, color="r")
plt.scatter (x_test, y_test, color="g")
plt.plot (x_test, model.predict (x_test), color="b")
plt.title ("Profit vs R&D spending")
plt.xlabel ("R&D spending")
plt.ylabel ("Profit")
plt.show ()