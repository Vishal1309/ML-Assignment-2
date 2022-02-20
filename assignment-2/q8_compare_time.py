import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import time
from os import path
import os
from tqdm import tqdm

np.random.seed(42)


if(path.exists('Plots/Question8/') == False):
    os.makedirs('Plots/Question8/')


print("For varying M with N fixed")
N = 6000
M = np.arange(50,5000,50)
gradient_descent_time = []
normal_time = []

for m in tqdm(M):
    np.random.seed(10)
    X = pd.DataFrame(np.random.randn(N, m))
    y = pd.Series(np.random.randn(N))

    LR = LinearRegression(fit_intercept=True)
    start_time = time.time()
    LR.fit_vectorised(X, y, 1,n_iter=10) 
    end_time = time.time()

    grad_train_time = end_time - start_time

    gradient_descent_time.append(grad_train_time)
    
    start_time = time.time()
    LR.fit_normal(X, y) 
    end_time = time.time()

    normal_train_time = end_time - start_time

    normal_time.append(normal_train_time)

fig = plt.figure()
plt.plot(M,gradient_descent_time,label="Gradient Descent")
plt.plot(M,normal_time,label = "Normal")
plt.title("Varying M with fixed N: Time required for gradient descent v/s normal")
plt.xlabel("M")
plt.ylabel("Time in seconds")
plt.legend()
fig.savefig('Plots/Question8/gradient_descent_vs_normal_varying_M.png')

fig = plt.figure()
plt.plot(M,gradient_descent_time,label="Gradient Descent")
plt.title("Varying M with fixed N: Time required for gradient descent")
plt.xlabel("M")
plt.ylabel("Time in seconds")
plt.legend()
fig.savefig('Plots/Question8/gradient_descent_varying_M.png')

fig = plt.figure()
plt.plot(M,normal_time,label = "Normal")
plt.title("Varying M with fixed N: Time required for normal")
plt.xlabel("M")
plt.ylabel("Time in seconds")
plt.legend()
fig.savefig('Plots/Question8/normal_varying_M.png')

print("For varying N with M fixed")
N = np.arange(5000,100000,5000)
M = 500
gradient_descent_time = []
normal_time = []

for n in tqdm(N):
    np.random.seed(10)

    X = pd.DataFrame(np.random.randn(n, M))
    y = pd.Series(np.random.randn(n))

    LR = LinearRegression(fit_intercept=True)
    start_time = time.time()
    LR.fit_vectorised(X, y, 1,n_iter=10) 
    end_time = time.time()

    grad_train_time = end_time - start_time

    gradient_descent_time.append(grad_train_time)
    
    start_time = time.time()
    LR.fit_normal(X, y) 
    end_time = time.time()

    normal_train_time = end_time - start_time

    normal_time.append(normal_train_time)

fig = plt.figure()
plt.plot(N,gradient_descent_time,label="Gradient Descent")
plt.plot(N,normal_time,label = "Normal")
plt.title("Varying N with fixed M: Time required for gradient descent v/s normal")
plt.xlabel("N")
plt.ylabel("Time in seconds")
plt.legend()
fig.savefig('Plots/Question8/gradient_descent_vs_normal_varying_N.png')

fig = plt.figure()
plt.plot(N,gradient_descent_time,label="Gradient Descent")
plt.title("Varying N with fixed M: Time required for gradient descent")
plt.xlabel("N")
plt.ylabel("Time in seconds")
plt.legend()
fig.savefig('Plots/Question8/gradient_descent_varying_N.png')

fig = plt.figure()
plt.plot(N,normal_time,label = "Normal")
plt.title("Varying N with fixed M: Time required for normal")
plt.xlabel("N")
plt.ylabel("Time in seconds")
plt.legend()
fig.savefig('Plots/Question8/normal_varying_N.png')


print("For varying iterations with M and N fixed")
print('Plots will take time to plot...')
N = 10
M = 5
np.random.seed(10)

X = pd.DataFrame(np.random.randn(N, M))
y = pd.Series(np.random.randn(N))

gradient_descent_time = []
normal_time = []
iterations = np.arange(500,20000,500)
for t in tqdm(iterations):

    LR = LinearRegression(fit_intercept=True)
    start_time = time.time()
    LR.fit_vectorised(X, y, 1,n_iter=t) 
    end_time = time.time()

    grad_train_time = end_time - start_time

    gradient_descent_time.append(grad_train_time)
    
    start_time = time.time()
    LR.fit_normal(X, y) 
    end_time = time.time()

    normal_train_time = end_time - start_time

    normal_time.append(normal_train_time)

fig = plt.figure()
plt.plot(iterations,gradient_descent_time,label="Gradient Descent")
plt.plot(iterations,normal_time,label = "Normal")
plt.title("Varying Iterations: Time required for gradient descent v/s normal")
plt.xlabel("Iterations")
plt.ylabel("Time in seconds")
plt.legend()
fig.savefig('Plots/Question8/gradient_descent_vs_normal_varying_iterations.png')











