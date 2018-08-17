"""
@author: venkatchidipudi
"""
# Regression Template

# importing the libraries
import numpy as np  #Contains the mathematical tools and includes any type of mathematical operations
import matplotlib.pyplot as plt  #Helps to plot the nice charts
import pandas as pd  #To import datasets and manage datasets

# Importing the datasets
dataset = pd.read_csv("Employee_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Fitting the Regression Model to the dataset
#Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, Y)

# Predecting a new result
Y_pred = regressor.predict(6.5)

# Visualising the Regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X,  regressor.predict(X), color = 'blue')
plt.title("Truth of Bluff (Regression Model)")
plt.xlabel("Employee Positions")
plt.ylabel("Salaries")
plt.show()

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(Y), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title("Truth of Bluff (Regression Model)")
plt.xlabel("Employee Positions")
plt.ylabel("Salaries")
plt.show()