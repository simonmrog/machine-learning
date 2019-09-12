#IMPORTING LIBRARIES
import numpy as np
import pandas as pd

#DATASET
filename = "Data.csv"
df = pd.read_csv (filename)
x = df.iloc[:, :-1].values
y = df.iloc[:, -1:].values

#CLEANING DATA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

imputer = SimpleImputer (missing_values = np.nan, strategy="mean")
x[:, 1:3] = imputer.fit_transform (x[:, 1:3])
labelencoder = LabelEncoder ()
x[:, 0] = labelencoder.fit_transform (x[:, 0])
onehotencoder = OneHotEncoder (categorical_features = [0])
x = onehotencoder.fit_transform (x).toarray ()
y[:, 0] = labelencoder.fit_transform (y[:, 0])

#SPLITTING DATASET
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=0)

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler ()
x_train = sc_x.fit_transform (x_train)
x_test = sc_x.transform (x_test)