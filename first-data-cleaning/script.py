#libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

#preparing data
missing_values = ["na", "n/a", "--"]
df = pd.read_csv ("data.csv", na_values = missing_values)

#data cleaning
i = 0
for row in df["OWN_OCCUPIED"]:
  try:
    int (row)
    df.loc[i, "OWN_OCCUPIED"] = np.nan
  except ValueError:
    pass
  i += 1

df["OWN_OCCUPIED"].fillna ("X", inplace=True)

  #creating vectors
x = df.iloc[:, :-1].values
y = df.iloc[:, -1:].values

imputer = SimpleImputer (missing_values = np.nan, strategy="mean", verbose=0)
x[:, 0:1] = imputer.fit_transform (x[:, 0:1])
y[:, 0:1] = imputer.fit_transform (y[:, 0:1])

labelencoder = LabelEncoder ()
x[:, 1] = labelencoder.fit_transform (x[:, 1])
x[:, 2] = labelencoder.fit_transform (x[:, 2])

onehotencoder = OneHotEncoder (categorical_features = [1, 2])
x = onehotencoder.fit_transform (x).toarray()

print (x)
