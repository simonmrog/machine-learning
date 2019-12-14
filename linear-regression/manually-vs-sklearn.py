import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_coefficients (X, Y):
    
    n = np.size(X)
    x_bar = np.mean (X)
    y_bar = np.mean (Y)
    
    xiyi = 0
    xi2 = 0
    
    for i in range (len (X)):
        xiyi += X[i]*Y[i]
        xi2 += X[i]**2
    
    b1 = (xiyi - n * x_bar * y_bar) / (xi2 - n * x_bar * x_bar)
    b0 = y_bar - b1 * x_bar
    
    return (b0, b1)
    
# preparing data
filename = "data.csv"
df = pd.read_csv (filename)
X = df.iloc [:, :-1].values
Y = df.iloc [:, -1:].values

#feature scaling
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler ()
#X = scaler.fit_transform (X)
#Y = scaler.transform (Y)

#train test splitting
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size=1/3)

#Fitting Simple Linear Regression to the training set
B = calculate_coefficients (X_train, Y_train)
Yp = B[0] + B[1] * X_test
#print ("y = %.2f + %.2fx" % (B[0], B[1]))

#fitting using sklearn
from sklearn.linear_model import LinearRegression
regression = LinearRegression ()
regression.fit (X_train, Y_train)
Ypred = regression.predict (X_test)

plt.scatter (X, Y)
print (Ypred-Yp) #difference between both predictions
plt.plot (X_test, Ypred, X_test, Yp)
plt.show ()
