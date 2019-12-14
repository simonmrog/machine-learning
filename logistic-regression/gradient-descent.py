# %%
import math
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:

  def __init__ (self, method="GD", learning_rate=0.001, max_iter=100, precision=0):
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
      DJ += (self.__linear_comb (theta, i) - self.y[i]) * weights[i]
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
      self.__update_theta ()
      #print (self.J[j])
      #evaluates convergency
      if (self.precision != 0 and abs (self.__J (self.theta) - self.J[j]) < self.precision):
        break

  def __calculate_theta (self):
    if (self.method == "GD"):
      self.__gradient_descent ()

  def fit (self, x_train, y_train):
      self.x = x_train
      self.y = y_train
      #print ("sum = ", sum ([i*i for i in y_train]) / (2.0 * len (y_train)))
      self.__calculate_theta ()

  def predict (self, x_test):
      x = np.append (np.ones ((len (x_test), 1)), x_test, axis=1)
      theta = np.array (self.theta)
      y = x @ theta
      return (y)

# %%loading admission predict dataset and scaling for gradient descent
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

# dataset = pd.read_csv ("../datasets/Admission_Predict_Ver1.1.csv")
# df = dataset.iloc [:, 1:]

# x = df[["GRE Score"]].to_numpy ().reshape (-1, 1)
# y = df.iloc[:, -1].to_numpy ()

# scaler = MinMaxScaler ()
# x = scaler.fit_transform (x)

# # %%train-test splitting
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=0)

# # %%
# from time import time
# start = time ()
# lr = LinearRegression (learning_rate=0.1, max_iter=600, precision=0)
# lr.fit (x_train, y_train)
# y_pred = lr.predict (x_train)
# print (lr.coefficients ())
# print (lr.mean_squared_error ())

# J = lr.cost_vector ()
# it = [i+1 for i in range (len (J))]
# final = time ()

# print (J[-1])
# print ("time:", final - start)

# plt.plot (it, J)
# plt.show ()

# plt.scatter (x, y, color="blue")
# plt.scatter (x_train, y_pred, color="red")
# plt.show ()

# %%
