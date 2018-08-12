"""
@author: venkatchidipudi
"""
#importing the libraries
import numpy as np  #Contains the mathematical tools and includes any type of mathematical operations
import matplotlib.pyplot as plt  #Helps to plot the nice charts
import pandas as pd  #To import datasets and manage datasets


#importing the datasets
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#Splitting the dataset into training dataset and test dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)


#Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predict the Test set result
Y_pred = regressor.predict(X_test)

#Visualise the Training set data
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'green')
plt.title("Salary vs Experience(Training set data)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary in Rupees")
plt.show()

#Visualise the Test set data
plt.scatter(X_test, Y_test, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'green')
plt.title("Salary and Experience(Test set data)")
plt.xlabel("Experience in Years")
plt.ylabel("Salary in Rupees")
plt.show()