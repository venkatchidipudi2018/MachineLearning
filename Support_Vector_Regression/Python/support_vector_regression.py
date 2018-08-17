"""
@author: venkatchidipudi
"""

# importing the libraries
import numpy as np  #Contains the mathematical tools and includes any type of mathematical operations
import matplotlib.pyplot as plt  #Helps to plot the nice charts
import pandas as pd  #To import datasets and manage datasets

# Importing the datasets
dataset = pd.read_csv("Employee_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, Y)

# Predecting a new result
Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# Visualising the SVR results
plt.scatter(X, Y, color = 'red')
plt.plot(X,  regressor.predict(X), color = 'blue')
plt.title("Truth of Bluff (SVR)")
plt.xlabel("Employee Positions")
plt.ylabel("Salaries")
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
#X_grid = np.arange(min(X), min(X), 0.1)
X_grid = X.reshape((len(X), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title("Truth or Bluff (SVR Model)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

