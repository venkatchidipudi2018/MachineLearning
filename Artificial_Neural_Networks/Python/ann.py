"""
@author: venkatchidipudi
"""

# Artificial Neural Networks

"""
Go to Anaconda prompt and install one by one

#1) Theano Library (Opensource and fast Numerical Computation purpose)
#pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
 
#2) TensorFlow (Opensource and fast Numerical Computation purpose)
#Follow the Anaconda installation steps in Tensor flow site like create virtual environment, activate and pip install the tensorflow
#https://www.tesorflow.org/versions/r0.11

#Keras Library (To built deep learning networks)
#pip install --upgrade keras

"""
# Part 1 - Data preprocessing

# importing the libraries
import numpy as np  #Contains the mathematical tools and includes any type of mathematical operations
import matplotlib.pyplot as plt  #Helps to plot the nice charts
import pandas as pd  #To import datasets and manage datasets

# importing the datasets
dataset = pd.read_csv("Customer_Details.csv")
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into training dataset and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Make ANN steps
#importing the libraries for ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

#Adding the input layer and the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

#Adding the output layer 
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training
classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
