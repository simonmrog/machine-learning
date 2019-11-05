# %%importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%loading the dataset
df = pd.read_csv ("social-network-ads.csv")
x = df[["Age", "EstimatedSalary"]].to_numpy ()
y = df["Purchased"].to_numpy ()


# %%train-test splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.25, random_state=0)

# %%feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler ()
x_train = scaler.fit_transform (x_train)
x_test = scaler.transform (x_test)

# %%fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
model = LogisticRegression (random_state=0)
model.fit (x_train, y_train)

# %%predicting results
y_hat = model.predict (x_test)

# %%evaluating the model using confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix (y_test, y_hat)
TN, FN, FP, TP = cm[0, 0], cm[1, 0], cm[0, 1], cm[1, 1]
acc = (TP + TN) / (TP + TN + FP + FN)
print ("acc: {0:.2f} %".format (acc * 100))

# %%visualizing train and test results
from matplotlib.colors import ListedColormap

def plot_results (x_set, y_set):
  x1_mesh = np.arange (start=x_set[:, 0].min () - 1, stop=x_set[:, 0].max () + 1, step=0.01)
  x2_mesh = np.arange (start=x_set[:, 1].min () - 1, stop=x_set[:, 1].max () + 1, step=0.01)

  x1, x2 = np.meshgrid (x1_mesh, x2_mesh)

  plt.contourf (x1, x2, model.predict (np.array ([x1.ravel (), x2.ravel ()]).T).reshape (x1.shape), alpha=0.7, cmap=ListedColormap (("red", "green")))
  plt.xlim (x1.min (), x1.max ())
  plt.ylim (x2.min (), x2.max ())

  for i, j in enumerate (np.unique (y_set)):
    plt.scatter (x_set[y_set==j, 0], x_set[y_set==j, 1], c = ListedColormap (("red", "green"))(j), label=j)

  plt.title ("Logistic Regression")
  plt.xlabel ("Age")
  plt.ylabel ("EstimatedSalary")
  plt.legend ()
  plt.show () 

plot_results (x_train, y_train)
plot_results (x_test, y_test)

# %%
