# %%importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error 

# %%loading the dataset
df = pd.read_csv ("teleCust1000t.csv")
df.head () 

# %%how many points we have of each class
df["custcat"].value_counts ()

# %%
x = df.iloc[:, :-1].to_numpy ()
y = df.iloc[:, -1:].to_numpy ().ravel ()

# %%feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler ()
x_sc = scaler.fit_transform (x)

# %%train-test splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x_sc, y, test_size=0.2, random_state=0)

# %%fitting the model to data
from sklearn.neighbors import KNeighborsClassifier
k = int (np.sqrt (len (x_train)))
print ("Optimal K value: ", k)
knn = KNeighborsClassifier (n_neighbors=k)
knn.fit (x_train, y_train)

# %%prediting results
y_hat = knn.predict (x_test)

# %%accuracy evaluations (Hamming distance)
print("Train set Accuracy: {0:.2f}%".format (accuracy_score(y_train, knn.predict(x_train)) * 100))
print("Test set Accuracy: {0:.2f}%".format (accuracy_score(y_test, y_hat) * 100))
print("Score: {0:.2f}%".format (knn.score (x_test, y_test) * 100))


# %%
