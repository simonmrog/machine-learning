#load the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#dataset
dates = pd.Series (["11/03/17", "23/12/16", "2019/3/12"])
data = pd.DataFrame (dates, columns=["date"])
data["date"].dtype

data["date"] = pd.to_datetime (data["date"], infer_datetime_format=True)
data

data["date"].dt.day