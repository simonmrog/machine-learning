#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading the dataset
dataset = pd.read_csv ("play-tennis.csv")
x_train = dataset.iloc[:, :-1].values
y_train = dataset.iloc[:, -1:].values
x_test = np.array (["Sunny", "Cool", "High", "Strong"]).reshape (1, -1)

#encoding categorical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_enc = LabelEncoder ()

for i in range (0, x_train.shape [1]):
  x_train[:, i] = label_enc.fit_transform (x_train[:, i])
  x_test[:, i] = label_enc.transform (x_test[:, i])

np.info (x_test)
np.info (x_train)

one_hot_enc = OneHotEncoder (categorical_features=[0, 1, 2, 3])
x_train = one_hot_enc.fit_transform (x_train).toarray ()
x_test = one_hot_enc.transform (x_test).toarray ()

#fitting the dataset to the model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier ()
model.fit (x_train, y_train)

#predicting results
y_pred = model.predict (x_test)
print (y_pred)