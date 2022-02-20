import numpy as np
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression
import pandas as pd
import os.path
from os import path
np.random.seed(10)  #Setting seed for reproducibility

x = np.array([i*np.pi/180 for i in range(60,300,4)])
y = 4*x + 7 + np.random.normal(0,3,len(x))

Y = pd.Series(y)
X = x.reshape(-1,1)
LR = LinearRegression(fit_intercept=False)
coeffs = []
degrees = []
for deg in range(1,7):
    poly = PolynomialFeatures(deg, include_bias=True)
    x_new = np.array([poly.transform(X[0])])
    for i in range(1,len(X)):
        x_new=np.concatenate((x_new,np.array([poly.transform(X[i])])))
    x_new=pd.DataFrame(x_new)
    LR.fit_vectorised(x_new, y, batch_size=7, lr_type='inverse')
    coeffs.append(np.log10(max(abs(LR.coef_))))
    degrees.append(deg)
    

if(path.exists('Plots/Question5/') == False):
                os.makedirs('Plots/Question5/')
plt.plot(degrees, coeffs)
plt.xlabel('Degree')
plt.ylabel('Max Absolute Value of thetha_i (log base 10 values)')
plt.savefig("Plots/Question5/theta_vs_deg.png")
