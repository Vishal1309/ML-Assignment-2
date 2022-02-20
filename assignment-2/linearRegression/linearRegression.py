import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.patches import Circle
import os
import imageio
np.random.seed(42)
import jax
import jax.numpy as jnp

# from torch import rand
# Import Autograd modules here

class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept 
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
        self.all_coef=pd.DataFrame([]) # Stores the thetas for every iteration (theta vectors appended)


    def fit_non_vectorised(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        self.all_coef=pd.DataFrame([])  #initializing all_coef as empty for each call of this function
        X_copy=X.copy()
        lr_=lr
        if (self.fit_intercept==True): #adding column of ones
            X_copy.insert(0,-1,1)
            X_copy.columns = np.arange(X_copy.shape[1])
        
        self.coef_=np.random.normal(0, 1, size=X_copy.shape[1]) #initializing thetas
        thetas=self.coef_
        self.all_coef[0]=thetas

        index=0 #for batch size

        for i in range (n_iter):            
            if (index>=X_copy.shape[0]):                    #if index crosses number of samples
                index=0

            if (lr_type=='inverse'):
                lr=lr_/(i+1)

            X_1=X_copy.iloc[index:index+batch_size,:]       #makes subset of X according to the batch size
            y_1=y[index:index+batch_size]
            y_hat=X_1.dot(thetas)                           #predictions

            for j in range(len(thetas)):                    #updating individual thetas
                thetas[j] = thetas[j] - lr*2*((y_1-y_hat)*(-X_1.iloc[:,j])).sum()
            
            index = index + batch_size                      #incrementing index by batch size
            self.all_coef[i+1]=thetas

        self.coef_ = thetas


    def fit_vectorised(self, X, y,batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        self.all_coef=pd.DataFrame([])
        X_copy=X.copy()
        if (self.fit_intercept==True):
            X_copy.insert(0,-1,1)
            X_copy.columns = np.arange(X_copy.shape[1])
        np.random.seed(42)
        self.coef_=np.random.normal(0, 1, size=X_copy.shape[1])
        thetas=self.coef_

        index=0
        self.all_coef[0]=thetas
        for i in range (n_iter):
            if (index>=X_copy.shape[0]):
                index=0

            if (lr_type=='inverse'):
                lr_1=lr/(i+1)
            else:
                lr_1=lr


            X_1=X_copy.iloc[index:index+batch_size,:]
            y_1=y[index:index+batch_size]
            y_hat=X_1.dot(thetas)

            thetas = thetas - (lr_1*2*((y_1-y_hat).T)@(-X_1.iloc[:])).T     #vectorized updation of thetas
            
            index = index + batch_size 
            self.all_coef[i+1]=thetas

        self.coef_ = thetas

        pass
    
    def mse_loss(self, coef_, X, y):                #mse function
        X_np=X.to_numpy()
        y_np=y.to_numpy()
        y_hat=jnp.dot(X_np, coef_)
        return jnp.mean(jnp.square(y_np - y_hat))



    def fit_autograd(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        self.all_coef=pd.DataFrame([])
        X_copy=X.copy()
        if (self.fit_intercept==True):
            X_copy.insert(0,-1,1)
            X_copy.columns = np.arange(X_copy.shape[1])
        np.random.seed(42)
        self.coef_=np.random.normal(0, 1, size=X_copy.shape[1])
        thetas=self.coef_

        index=0 
        grads = jax.grad(self.mse_loss)             #computes gradients
        for i in range (n_iter):
            if (index>=X_copy.shape[0]):
                index=0

            if (lr_type=='inverse'):
                lr_1=lr/(i+1)
            else:
                lr_1=lr


            X_1=X_copy.iloc[index:index+batch_size,:]
            y_1=y[index:index+batch_size]

            thetas = thetas - (lr_1*grads(thetas, X_1, y_1))    #updates thetas using gradients calculated by inbuilt function
            
            index = index + batch_size 

        self.coef_ = thetas

        pass

    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''
        if np.linalg.det(X.T @ X) == 0:         #returns error in case where dataset suffers multicolinearity
            print("Matrix Singularity Error on Normal Equation")
        else:
            self.coef_ = np.linalg.inv((X.T @ X)) @ (X.T @ y)

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        X_copy=X.copy()
        if (len(self.coef_)!=X_copy.shape[1]):      #adding column of 1 for bias
            X_copy.insert(0,-1,1)
            X_copy.columns = np.arange(X_copy.shape[1])
        return X_copy @ self.coef_

        pass

    def plot_surface(self, X, y, t_0, t_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS #pd Series of all theta_0
        :param t_1: Value of theta_1 for which to indicate RSS #pd Series of all theta_1

        :return matplotlib figure plotting RSS
        """
        filenames = []
        fig=plt.figure()
        c = np.arange(-0.5, 6, 0.05)
        m = np.arange(-0.5,6,0.05)
        c,m = np.meshgrid(c,m)
        axes = fig.gca(projection ='3d')
        errs=[]
        for C in range(c.shape[0]):
            e = []
            for M in range(m.shape[1]):
                e.append(np.sum(np.square(y - (c[C][M] + X*m[C][M]))))
            errs.append(e)
        errs=np.array(errs)
        plot=axes.plot_surface(c,m,errs, cmap = cm.coolwarm)
        axes.view_init(20, 120)
        axes.set_xlabel(r'$\theta_0$')
        axes.set_ylabel(r'$\theta_1$')
        fig.colorbar(plot, shrink=0.8)
        for iter,(t0,t1) in enumerate(zip(t_0,t_1)):
            
            y_hat=t0 + X*t1
            err=np.sum(np.square(y-y_hat))
            axes.set_title(f'Error = {err:.2f}')
            print(iter,t0,t1,err)
            
            p = Circle((t0, t1), 0.05,color='black')
            axes.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=err, zdir="z")
            filename = f'{iter}.png'
            filenames.append(filename)
            plt.savefig(filename)

        with imageio.get_writer('surface.gif',mode = 'I',fps=2) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

            
        for filename in set(filenames):
            os.remove(filename)

        pass

    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """

        filenames=[]
        X_np = np.array(X)
        y_np = np.array(y)
        for iter,(t0,t1) in enumerate(zip(t_0,t_1)):
            fig=plt.figure()
            plt.ylim(0,35)
            plt.xlabel("Feature")
            plt.ylabel("Output")
            plt.title(f"y = {t1:0.2f}x + {t0:0.2f}")
            plt.scatter(X_np,y_np)
            
            y_hat=t0 + X*t1
            err=np.sum(np.square(y-y_hat))
            plt.plot(X_np, y_hat)
            filename = f'{iter}.png'
            filenames.append(filename)
            plt.savefig(filename)
            
        with imageio.get_writer('line_fit.gif',mode = 'I',fps=2) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        for filename in set(filenames):
            os.remove(filename)
        pass

    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """
        filenames = []
        fig=plt.figure()
        c = np.arange(-5, 6, 0.1)
        m = np.arange(-5,6,0.1)
        c,m = np.meshgrid(c,m)

        errs=[]
        for C in range(c.shape[0]):
            e = []
            for M in range(m.shape[1]):
                e.append(np.sum(np.square(y - (c[C][M] + X*m[C][M]))))
            errs.append(e)
        errs=np.array(errs)

        plot=plt.contourf(c,m,errs)
        fig.colorbar(plot, shrink=0.8)
        plt.xlabel("c")
        plt.ylabel("m")
        plt.title('Contour plot')

        for iter,(t0,t1) in enumerate(zip(t_0,t_1)):
            
            y_hat=t0 + X*t1
            err=np.sum(np.square(y-y_hat))
            # print(iter,t0,t1,err)
            plt.scatter(t0, t1,s=28, marker ='3', color = 'red')
            filename = f'{iter}.png'
            filenames.append(filename)
            plt.savefig(filename)

        with imageio.get_writer('contour.gif',mode = 'I',fps=2) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        for filename in set(filenames):
            os.remove(filename)
        
        pass
