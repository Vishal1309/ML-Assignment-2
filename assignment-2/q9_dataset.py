import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

x = np.array([i*np.pi/180 for i in range(60,300,4)]) 
X = np.array([x, 2*x])
X = X.T
np.random.seed(10) 
y = 4*x + 7 + np.random.normal(0,3,len(x)) 

X = pd.DataFrame(X)
y = pd.Series(y)


# Gradient Descent
LR = LinearRegression(fit_intercept=True)
LR.fit_vectorised(X, y,1)
y_hat = LR.predict(X)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))

print("Gradient Descent functions well!")

# Normal equation method
LR1 = LinearRegression(fit_intercept=True)
LR1.fit_normal(X, y)
# should give an error message now

# Gradient Descent works on multicollinear data also, unlike normal equation method because of matrix singularity of (X.T @ T)
