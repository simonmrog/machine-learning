#%%importing libraries
import pandas as pd
import seaborn as sns

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

#%%loading the dataset
dataframe = pd.read_csv ("winequality-red.csv")
x = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]


#%%creating pipeline
steps = [("scaler", StandardScaler ()), ("SVM", SVC ())]

from sklearn.pipeline import Pipeline
pipeline = Pipeline (steps)

#%%train-test splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=30, stratify=y)


#%%
parameteres = {'SVM__C':[0.001,0.1,10,100,10e5], 'SVM__gamma':[0.1,0.01]}
#%%
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipeline, param_grid=parameteres, cv=5)
grid.fit (x_train, y_train)
#%%
print (grid.score (x_test, y_test))

#%%
