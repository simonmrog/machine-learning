# %%
import numpy as np

# %%
class LinearRegression:

  def __init__ (self):
    self.b = []

  def fit (self, x_train, y_train):
    x = np.append (np.ones ((x_train.shape [0], 1)), x_train, axis=1)
    self.b = np.linalg.inv (x.T @ x) @ x.T @ y_train

  def predict (self, x_test):
    x = np.append (np.ones ((x_test.shape [0], 1)), x_test, axis=1)
    return (x @ self.b)

  def score (self, x_test, y_test):
    u = ((y_test - self.predict (x_test)) ** 2).sum ()
    v = ((y_test - y_test.mean()) ** 2).sum()
    return (1 - u/v)


# %%preparing the dataset
from sklearn import datasets

iris = datasets.load_iris ().data
x = iris [:, :-1]
y = iris [:, -1]

# %%train-test splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=0)

# %%fitting the model to the data

model = LinearRegression ()
model.fit (x_train, y_train)
y_pred = model.predict (x_test)
acc = model.score (x_test, y_test)
print ("my acc: {0:.2f}".format (acc))

# %%fitting data to the sklearn linear regressor
from sklearn import linear_model
sk_model = linear_model.LinearRegression ()
sk_model.fit (x_train, y_train)
sk_y_pred = sk_model.predict (x_test)
sk_acc = sk_model.score (x_test, y_test)
print ("sklearn acc: {0:.2f}".format (sk_acc))

# %%comparing results
print (y_pred - sk_y_pred)

# %%
