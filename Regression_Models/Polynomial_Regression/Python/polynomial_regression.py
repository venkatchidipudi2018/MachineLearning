"""
@author: venkatchidipudi
"""
#Polinomial Regression
#importing the libraries
import numpy as np  #Contains the mathematical tools and includes any type of mathematical operations
import matplotlib.pyplot as plt  #Helps to plot the nice charts
import pandas as pd  #To import datasets and manage datasets

#Importing the datasets
dataset = pd.read_csv("Employee_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)

#Visualising the Linear Regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title("Linear Regression of Employee Salary")
plt.xlabel("Employee Positions")
plt.ylabel("Salaries")
plt.show()

#Visualising the Polynomial Regression results
X_grid = np.arange(min(X), max(Y), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title("Polynomial Regression of Employee Salary")
plt.xlabel("Employee Positions")
plt.ylabel("Salaries")
plt.show()

#Predecting a new result with Linear Regression
lin_reg.predict(6.5)

#Predecting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform(6.5))



