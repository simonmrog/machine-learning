import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:

  b0 = b1 = 0

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


def main ():

  x_train = np.array ([1, 2, 3, 4, 5])
  y_train = np.array ([2, 4, 5, 6, 7])

  x_test = np.array ([1, 2, 3, 4, 5])

  regressor = LinearRegression ()
  regressor.fit (x_train, y_train)
  regressor.coef ()
  y_pred = regressor.predict (x1)

  plt.scatter (x_train, y_train)
  plt.plot (x_test, y_pred)


if __name__ == "__main__":
  main ()