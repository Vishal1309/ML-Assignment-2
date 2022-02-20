import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression
import os.path
from os import path

if(path.exists('Plots/Question6/') == False):
    os.makedirs('Plots/Question6/')

fig=plt.figure()

for N in range(40,200,20):
    x = np.array([i*np.pi/180 for i in range(N,N*5,4)])
    np.random.seed(10)  #Setting seed for reproducibility
    y = 4*x + 7 + np.random.normal(0,3,len(x))

    Y = pd.Series(y)
    X = x.reshape(-1,1)
    LR = LinearRegression(fit_intercept=False)
    coeffs = []
    degrees = []
    list=[1,3,5,7,9]
    for deg in list:
        poly = PolynomialFeatures(deg, include_bias=True)
        x_new = np.array([poly.transform(X[0])])
        for i in range(1,len(X)):
            x_new=np.concatenate((x_new,np.array([poly.transform(X[i])])))
        x_new=pd.DataFrame(x_new)
        LR.fit_vectorised(x_new, y, batch_size=7, lr_type='inverse')
        coeffs.append(np.log10(max(abs(LR.coef_))))
        degrees.append(deg)


    plt.plot(degrees, coeffs,label=N)
plt.xlabel('Degree')
plt.ylabel('Max Absolute Value of thetha_i')
plt.legend()
plt.savefig("Plots/Question6/theta_vs_deg_for_Ns.png")
