# ES654-2022 Assignment 2

*Aadesh Desai, Eshan Gujarathu, Vishal Soni* - *19110116, 19110082, 19110207*

------

# Dataset

The columns are dependent on each other as follows:

    > feature1 = np.array([i*np.pi/180 for i in range(60, 300, 4)])

    > feature2 = feature1*2

# Dataset on Gradiant descent Method:

    > RMSE: 7.055655852484071

    > MAE: 6.525090976490666 

 Gradient Descent works for the multicolinearity dataset. The values of RMSE and MAE is high as the features are dependent.

# Dataset on Normal method:

    > Error, X transpose * X is not invertable
