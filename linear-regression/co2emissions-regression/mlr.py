#%%importing the libraries
import pandas as pd
import numpy as np

# %%loading the dataset
dataset = pd.read_csv ("co2emissions.csv")
x = dataset.iloc [:, 0:-1].values
y = dataset.iloc [:, -1].values

# %%train-test-splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.15)

# %%fitting the model to data
from sklearn.linear_model import LinearRegression
model = LinearRegression ()
model.fit (x_train, y_train)

# %%
y_pred = model.predict (x_test)
acc = model.score (x_test, y_test)
print ("{:.2f}%".format (acc * 100))
print (model.predict ([[3.66, 6, 11]]))

# %%importing pickle
import pickle
pickle.dump (model, open ("model.pickle", "wb"))