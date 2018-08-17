"""
@author: venkatchidipudi
"""
# Random Forest Regression model

# importing the libraries
import numpy as np  #Contains the mathematical tools and includes any type of mathematical operations
import matplotlib.pyplot as plt  #Helps to plot the nice charts
import pandas as pd  #To import datasets and manage datasets

# Importing the datasets
dataset = pd.read_csv("Employee_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Fitting the Random forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, Y)

# Predicting a new result
y_pred = regressor.predict(6.5)

#Visualising the regrssion result (for highest resolution and smoother)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Randon Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()