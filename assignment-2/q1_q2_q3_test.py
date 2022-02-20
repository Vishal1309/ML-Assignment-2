
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))
print(X.shape)


for fit_intercept in [False, True]:
    for lrtype in ['inverse','constant']:
        for batchsize in [1,15,30]:
            LR = LinearRegression(fit_intercept=fit_intercept)
            LR.fit_non_vectorised(X, y, batchsize, lr_type=lrtype) # here you can use fit_non_vectorised / fit_autograd methods
            y_hat = LR.predict(X)

            print('Fit intercept=',fit_intercept,', lr_type=',lrtype,', Batch size=',batchsize,', RMSE: ', rmse(y_hat, y))
            print('Fit intercept=',fit_intercept,', lr_type=',lrtype,', Batch size=',batchsize,', MAE: ', mae(y_hat, y))
print("---------------------------")
for fit_intercept in [False, True]:
    for lrtype in ['inverse','constant']:
        for batchsize in [1,15,30]:
            LR = LinearRegression(fit_intercept=fit_intercept)
            LR.fit_vectorised(X, y, batchsize, lr_type=lrtype) # here you can use fit_non_vectorised / fit_autograd methods
            y_hat = LR.predict(X)

            print('Fit intercept=',fit_intercept,', lr_type=',lrtype,', Batch size=',batchsize,', RMSE: ', rmse(y_hat, y))
            print('Fit intercept=',fit_intercept,', lr_type=',lrtype,', Batch size=',batchsize,', MAE: ', mae(y_hat, y))
print("---------------------------")
for fit_intercept in [False, True]:
    for lrtype in ['inverse','constant']:
        for batchsize in [1,15,30]:
            LR = LinearRegression(fit_intercept=fit_intercept)
            LR.fit_autograd(X, y, batchsize, lr_type=lrtype) # here you can use fit_non_vectorised / fit_autograd methods
            y_hat = LR.predict(X)

            print('Fit intercept=',fit_intercept,', lr_type=',lrtype,', Batch size=',batchsize,', RMSE: ', rmse(y_hat, y))
            print('Fit inept=',fit_intercept,', lr_type=',lrtype,', Batch size=',batchsize,', MAE: ', mae(y_hat, y))