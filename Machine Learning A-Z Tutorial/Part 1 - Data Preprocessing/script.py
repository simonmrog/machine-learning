import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

filename = "Data.csv"
df = pd.read_csv (filename)
x = df.iloc[:, :-1].values
y = df.iloc[:, -1:].values

imputer = SimpleImputer (missing_values = np.nan, strategy="mean")
x[:, 1:3] = imputer.fit_transform (x[:, 1:3])

labelencoder = LabelEncoder ()
x[:, 0] = labelencoder.fit_transform (x[:, 0])
onehotencoder = OneHotEncoder (categorical_features = [0])
x = onehotencoder.fit_transform (x).toarray ()

y[:, 0] = labelencoder.fit_transform (y[:, 0])