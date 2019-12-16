# %%
import math
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionS:

  def __init__ (self, method="GD", learning_rate=0.01, max_iter=100, precision=0):
    self.method = method
    self.theta = []
    self.J = []

    self.learning_rate = learning_rate
    self.max_iter = max_iter
    self.precision = precision

  def coefficients (self):
    return (self.theta)

  def cost_vector (self):
    return (self.J)

  def mean_squared_error (self):
    return (self.J[-1])

  def __linear_comb (self, theta, i):
    lc = 0
    for j in range (len (self.x[i, :])):
      lc += theta[j+1] * self.x[i, j] 
    return (lc + theta[0])

  def __sigmoid (self, theta, i):
    power = self.__linear_comb (theta, i) * (-1)
    return (1.0 / (1 + math.exp (power)))

  def __J (self, theta):
    J = 0
    for i in range (len (self.x)):
      log1 = math.log (self.__sigmoid (theta, i))
      log2 = math.log (1 - self.__sigmoid (theta, i))
      J += (self.y[i] * log1  + (1 - self.__sigmoid (theta, i)) * log2)
    return ((-1.0 / len (self.x)) * J)

  def __derivative_J (self, theta, k):
    DJ = 0
    if (k == 0):
      weights = [1 for i in range (len (self.x))]
    else:
      weights = self.x[:, k-1]

    for i in range (len (self.x)):
      h_i = self.__sigmoid (theta, i)
      DJ += (h_i - self.y[i]) * weights[i]
    return ((1.0 / len (self.x)) * DJ)

  def __update_theta (self):
    actual_theta = self.theta
    for j in range (len (self.theta)):
      self.theta[j] = actual_theta[j] - self.learning_rate * self.__derivative_J (actual_theta , j)

  def __gradient_descent (self):
    #initializing theta values
    for i in x_train[0, :]:
      self.theta.append (0)
    self.theta.append (0)

    for j in range (self.max_iter): 
      self.J.append (self.__J (self.theta))
      # print ("j={}, J(theta)={}".format (j, self.J))
      self.__update_theta ()
      #evaluates convergency
      if (self.precision != 0 and abs (self.__J (self.theta) - self.J[j]) < self.precision):
        break

  def __calculate_theta (self):
    if (self.method == "GD"):
      self.__gradient_descent ()

  def fit (self, x_train, y_train):
      self.x = x_train
      self.y = y_train
      self.__calculate_theta ()

  def predict_proba (self, x_test):
      x = np.append (np.ones ((len (x_test), 1)), x_test, axis=1)
      theta = np.array (self.theta)
      y = []
      for xi in x:
        power = (xi @ theta) * (-1)
        y.append (1.0 / (1 + math.exp (power)))
      y = np.array (y)
      return (y)

  def predict (self, x_test):
    y = self.predict_proba (x_test)
    return (y >= 0.5)

# %%loading admission predict dataset and scaling for gradient descent
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

df = sns.load_dataset ("iris")
x = df.iloc[:100, 2].to_numpy ()
y = df.iloc[:100, -1].to_numpy ()

#handling categorical variable
encoder = LabelEncoder ()
y = encoder.fit_transform (y)

#scaling the dataset
scaler = MinMaxScaler ()
# x = scaler.fit_transform (x)

# %%train-test splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x.reshape (-1, 1), y, test_size=0.2, random_state=0)

# %%using sklearn logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

lr = LogisticRegression (solver="lbfgs")
lr.fit (x_train, y_train)
y_pred = lr.predict_proba (x_test)[:, 1]
# print ("sklearn probabilities:", y_pred)
print ("sklearn coefficients: [{}, {}]".format (lr.intercept_[0], lr.coef_[0][0]))
mse = mean_squared_error (y_test, y_pred)
print ("sklearn MSE: {}".format (mse))

plt.scatter (x_test.reshape (-1, 1), y_test, color="blue")
plt.scatter (x_test.reshape (-1, 1), y_pred, color="red")
plt.show ()

# %%
from time import time
# start = time ()
model = LogisticRegressionS (learning_rate=1.6, max_iter=100, precision=0)
model.fit (x_train, y_train)
y_pred = model.predict (x_test)
y_pred = model.predict_proba (x_test)
print ("model coefficients:", model.coefficients ())
# print ("model probabilities:", y_pred)
mse = mean_squared_error (y_test, y_pred)
print ("model MSE:", mse)

J = model.cost_vector ()
it = [i+1 for i in range (len (J))]
# final = time ()

# print (J[-1])
# # print ("time:", final - start)

plt.scatter (x_test.reshape (-1, 1), y_test, color="blue")
plt.scatter (x_test.reshape (-1, 1), y_pred, color="red")
plt.show ()
plt.plot (it, J)
plt.show ()

# %%
import pandas as pd

data=pd.read_csv("../datasets/social-network-ads.csv",delimiter=",")
x=data.iloc[:,2:-1].values
y=data.iloc[:,-1:].values
y_reshaped = y.reshape(-1, )

def normalization_max_min (x):
  xmin = np.min(x, axis=0)
  xmax = np.max(x, axis=0)
  x_norm = (x-xmin)/(xmax-xmin)
  return(x_norm)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (normalization_max_min(x), y_reshaped, test_size=0.2, random_state=0)

x1_train_0=[]
x2_train_0=[]
x1_train_1=[]
x2_train_1=[]
x1=[]
for i in range (len(y_train)):
  if y_train[i]==0:
    x1_train_0.append(x_train[i][0])
    x2_train_0.append(x_train[i][1])
  else:
    x1_train_1.append(x_train[i][0])
    x2_train_1.append(x_train[i][1])
  x1.append(x_train[i][0])

# %%testing the model
model = LogisticRegressionS (learning_rate=5.5, max_iter=811, precision=1e-9)
model.fit (x_train, y_train)
theta = model.coefficients ()
J = model.cost_vector ()
print ("J:", J[-1])

cost_vs_cycles=plt.plot([i for i in range (811)], J)
plt.show ()

plt.scatter (x1_train_0, x2_train_0)
plt.scatter (x1_train_1, x2_train_1)
plt.plot(x1,-(np.multiply (theta[1], x1) + theta[0]) / theta[2])
plt.show ()

# %%sklearn model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
sklearn_model = LogisticRegression ()
sklearn_model.fit (x_train, y_train)
theta = [sklearn_model.intercept_[0]]
theta.append (sklearn_model.coef_[0][0])
theta.append (sklearn_model.coef_[0][1])
print ("Theta:", theta_final)
y_pred = sklearn_model.predict_proba (x_test)[:, 1]
mse = mean_squared_error (y_test, y_pred)
print ("J:", mse)

plt.scatter (x1_train_0, x2_train_0)
plt.scatter (x1_train_1, x2_train_1)
plt.plot(x1,-(np.multiply (theta[1], x1) + theta[0]) / theta[2])
plt.show ()
# %%
