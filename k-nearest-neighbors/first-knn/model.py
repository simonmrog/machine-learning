# %%importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error 

# %%loading the data
dataset = pd.read_csv ("iris.csv")
x = dataset[["SEPAL_LENGTH", "SEPAL_WIDTH"]].to_numpy ()
y = dataset["PETAL_LENGTH"].to_numpy ()
dataset.head ()

# %%plotting the data points
plt.scatter (x[:, 0], x[:, 1], color="blue")

# %%train-test-split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2)

# %%feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler (feature_range=(0, 1))
x_train_scaled = scaler.fit_transform (x_train)
x_test_scaled = scaler.transform (x_test)

# %%finding the optimal K-value
rmse = []
K = 0
for k in range (1, 20, 1):
  model = KNeighborsRegressor (n_neighbors=k)
  model.fit (x_train_scaled, y_train)
  y_pred = model.predict (x_test_scaled)
  err = np.sqrt (mean_squared_error (y_test, y_pred))
  if (all (e > err for e in rmse)):
    K = k
  rmse.append (err)
  print ("RMSE for k={} is {}: ".format (k, err))

print ("K optimal is ", K)


# %%plotting the rmse against k
elbow_curve = pd.DataFrame (rmse)
elbow_curve.plot ()

# %%train the model with optimal k-value
K=7
model = KNeighborsRegressor (n_neighbors=K)
model.fit (x_train_scaled, y_train)
y_pred = model.predict (x_test_scaled)
acc = model.score (x_test_scaled, y_test)
print ("Accuracy: {}%".format (acc * 100))

# %%plotting some results
plt.scatter (dataset["SEPAL_LENGTH"], dataset["PETAL_LENGTH"], color="blue")
plt.scatter (x_test[:, 0], y_pred, color="red")
plt.scatter (x_test[:, 0], y_test, color="green")
plt.show ()