# ES654-2022 Assignment 2

*Aadesh Desai, Eshan Gujarathu, Vishal Soni* - *19110116, 19110082, 19110207*


## Question 5

## Magnitude of theta v/s degree when our linear regression is fit against varying polynomial degrees:
![](/assignment-2/Plots/Question5/theta_vs_deg.png)

## Observations
1. From this, we can conclude that as the degree of the polynomial increases, the absolute value of the highest coefficient increases when we try to fit our linear regressor on the dataset.
2. This is because when we increase the degree, there will be high power attributes in my X. Lets take an example, if my degree is 9, there will be attributes x1 and x1^9 in my dataset. So if x1 is greater than 1, x1^9 will be very large as compared to x1, so the theta1 for x1 will have to be very large for compensating this value. Thus maximum theta value increases with degree.
3. When we move beyond degree 5, the absolute of the highest coefficient goes beyond the limit of the data structure and is too large.
