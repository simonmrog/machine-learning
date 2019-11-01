# %%importing libraries
import pandas as pd
from k_nearest_neighbors import KNNRegressor
import matplotlib.pyplot as plt

# %%loading the dataset
df = pd.read_csv ("iris.csv")
x = df[["SEPAL_LENGTH", "SEPAL_WIDTH"]].to_numpy ()
y = df["PETAL_LENGTH"].to_numpy ()
df.head ()

# %%training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=0)

# %%fitting model to data
knn_reg = KNNRegressor (k=3)
knn_reg.fit (x_train, y_train)
y_pred = knn_reg.predict (x_test)
print (knn_reg.score (x_test, y_test))

# %%fitting with the sklearn model
from sklearn.neighbors import KNeighborsRegressor
sk_reg = KNeighborsRegressor (n_neighbors=3)
sk_reg.fit (x_train, y_train)
y_pred_sk = sk_reg.predict (x_test)
print (sk_reg.score (x_test, y_test))

# %%plotting the difference respect the sklearn model
plt.scatter (x_test[:, 0], y_test, color="blue")
plt.scatter (x_test[:, 0], y_pred_sk, color="red")
plt.scatter (x_test[:, 0], y_pred, color="green")
plt.show ()

# %%
