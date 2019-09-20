import math
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:

  b0 = b1 = r = 0

  def coef (self):
    return (self.b0, self.b1)

  def fit (self, x, y):

    n = len (x)
    xiyi = 0
    xi2 = 0

    for i in range (n):
      xiyi += x[i] * y[i]
      xi2 += x[i] ** 2

    self.b1 = (xiyi - n * x.mean () * y.mean ()) / (xi2 - n * x.mean () ** 2)
    self.b0 = y.mean () - self.b1 * x.mean ()

  def predict (self, x):

    y = self.b0 + self.b1 * x
    return (y)

  def score (self, x, y):
  
    y_pred = self.predict (x)

    var_y_pred = ((y_pred - y_pred.mean ()) * (y_pred - y_pred.mean ())).sum ()
    var_y = ((y - y.mean ()) * (y - y.mean ())).sum ()
    r = math.sqrt (var_y_pred / var_y)
    return (r)

def main ():

  x_train = np.array ([1, 2, 3, 4, 5])
  y_train = np.array ([2, 4, 5, 6, 7])

  x_test = np.array ([1, 2, 3, 4, 5])

  regressor = LinearRegression ()
  regressor.fit (x_train, y_train)
  regressor.coef ()
  y_pred = regressor.predict (x_test)
  r2 = regressor.score (x_train, y_train)
  print (r2)
  plt.scatter (x_train, y_train)
  plt.plot (x_test, y_pred)


if __name__ == "__main__":
  main ()