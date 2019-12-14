# %%
import statistics
import numpy as np
from sklearn.metrics import r2_score

class KNNRegressor:
  def __init__ (self, k):
    self.k = k
    self.x = []
    self.y = []
    self.dist = []
    print ("KNNRegressor with K =", self.k)

  def __calculate_distances (self, x_test):
    for i in range (len (x_test)):    
      vec = []
      for j in range (len (self.x)):
        l2norm = 0
        for k in range (len (self.x[i])):
          l2norm += (x_test [i,k] - self.x[j][k]) ** 2
        
        vec.append({
          "norm": np.sqrt (l2norm),
          "value": self.y[j]
        })
      self.dist.append (vec)

  def __sort_key (self, data_point):
    return (data_point ["norm"])

  def __sort_k_nearest_neighbors (self):
    for i in range (len (self.dist)):
      self.dist[i].sort (key=self.__sort_key)

  def __mean (self):
    means = []
    for i in range (len (self.dist)):
      mean = 0
      for j in range (self.k):
        mean += self.dist[i][j]["value"]
      means.append (mean / self.k)
    return (means)

  def fit (self, x_train, y_train):
    for i in range (len (x_train)):
      self.x.append (x_train[i, :])
      self.y.append (y_train[i])

  def predict (self, x_test):

    self.__calculate_distances (x_test)
    self.__sort_k_nearest_neighbors ()
    return (self.__mean ())

  def score (self, x_test, y_test):
    self.dist = []
    pred = self.predict (x_test)
    var_hat = statistics.variance (pred, xbar=None)
    var = statistics.variance (y_test, xbar=None)
    r2 = var_hat / var
    r2 = r2_score (y_test, pred)
    return (r2)

class KNNClassifier:
  def __init__(self, k):
    self.k = k
    print ("KNNClassifier with K =", k)

# %%
