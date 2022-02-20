import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from os import path
import os

N=60
x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)
y = 4*x + 7 + np.random.normal(0,3,len(x)) 


y=pd.Series(y)
LR = LinearRegression(fit_intercept=True)
LR.fit_vectorised(pd.DataFrame(x), y, batch_size=1, n_iter=10)
LR.plot_surface(pd.Series(x),y,LR.all_coef.iloc[0],LR.all_coef.iloc[1])
LR.plot_line_fit(pd.Series(x),y,LR.all_coef.iloc[0],LR.all_coef.iloc[1])
LR.plot_contour(pd.Series(x),y,LR.all_coef.iloc[0],LR.all_coef.iloc[1])
