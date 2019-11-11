#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: baljeetsingh
"""
import math;

from sklearn.datasets import load_boston
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd


class LinearRegression():
    
#    """ Class Intializer""""
    def __init__(self, X, y, alpha=0.03, n_iter=2500):

        self.alpha = alpha
        self.n_iter = n_iter
        self.n_samples = len(y)
        self.n_features = np.size(X, 1)
        self.X = np.hstack((np.ones(
            (self.n_samples, 1)), (X - np.mean(X, 0)) / np.std(X, 0)))
        self.y = y[:, np.newaxis]
        self.params = np.zeros((self.n_features + 1, 1))
        self.coef_ = None
        self.intercept_ = None

#    """Gradient Descent""""
    def gradient_decent(self):

        for i in range(self.n_iter):
            gradient = self.X.T @ (self.X @ self.params - self.y)
            learning_rate =self.alpha/self.n_samples
            self.params -= learning_rate * gradient
#        print(self.params)
        self.intercept_ = self.params[0]
        self.coef_ = self.params[1:]

        return self

#    """Nesterov gradient descent""""
    def nesterov_gradient_descent(self,gamma=0.9, epsilon=1e-4):
      
        lr = self.alpha/self.n_samples;
        v_a = np.zeros((self.n_features + 1, 1))
        
        
        for i in range(self.n_iter):
            a_tmp=self.params - gamma*v_a
            gradient = self.X.T @ (self.X @ a_tmp - self.y)
            v_a = lr * gradient
            self.params -= v_a
            
        self.intercept_ = self.params[0]
        self.coef_ = self.params[1:]

        return self
    
#    """Shuffling for batch preparation""""
    def shuffle(self,x, y):
        s = np.arange(len(x))
        np.random.shuffle(s)
        return x[s], y[s]

#    """Momentum Gradient Descent""""
    def momentum_gradient_descent(self,momentum=0.9,epsilon=1e-4, batch_size=10):
#        print(self.X)
        if batch_size == 0: batch_size = self.X.shape[0]
        lr = self.alpha/self.n_samples;
        prev_grad_a = np.zeros((self.n_features + 1, 1))
        for i in range(self.n_iter):
            X, Y = self.shuffle(self.X, self.y)
            X = X[:batch_size]
            Y = Y[:batch_size]
#            print(self.params.shape)
            gradient = X.T @ (X @ self.params - Y)
            self.params -= lr * gradient + momentum * prev_grad_a
            prev_grad_a = lr * gradient + momentum * prev_grad_a
            
        self.intercept_ = self.params[0]
        self.coef_ = self.params[1:]

        return self
    

#    """Stochastic Gradient Descent""""
    def stochastic_gradient_descent(self,epsilon=1e-4, batch_size=200):
        lr = self.alpha/self.n_samples;
        if batch_size == 0: batch_size = self.X.shape[0]
   
        for i in range(self.n_iter):
            X, Y = self.shuffle(self.X, self.y)
            X = X[:batch_size]
            Y = Y[:batch_size]

            gradient = X.T @ (X @ self.params - Y)
            self.params -= lr * gradient 
            
        self.intercept_ = self.params[0]
        self.coef_ = self.params[1:]

        return self
    
        
#    """ Finding RMSE""""
    def rmse_finder(self, X=None, y=None):

        if X is None:
            X = self.X
        else:
            n_samples = np.size(X, 0)
            X = np.hstack((np.ones(
                (n_samples, 1)), (X - np.mean(X, 0)) / np.std(X, 0)))

        if y is None:
            y = self.y
        else:
            y = y[:, np.newaxis]
        
            
        y_pred = X @ self.params
        
        rss=((y-y_pred)**2).sum() 
        
        rmse_finder =  np.sqrt(rss/y.shape[0])

        return rmse_finder #Root mean square error

    def predict(self, X):
        n_samples = np.size(X, 0)
        y = np.hstack((np.ones((n_samples, 1)), (X-np.mean(X, 0)) \
                            / np.std(X, 0))) @ self.params
        return y

    def get_params(self):

        return self.params
    

class linear_adagrad:
    def split(X, y, ratio, randomness=5):
        # Defining the randomness using the seed function
        np.random.seed(randomness)
        rows = len(y)
        indices = np.random.permutation(rows)
        marker = int(np.floor(ratio * rows))
        index_train = indices[: marker]
        index_test = indices[marker:]
    
        # Splitting by allocation of dataframes
        X_Train = X.iloc[index_train]
        X_Test = X.iloc[index_test]
        y_train = y.iloc[index_train]
        y_test = y.iloc[index_test]
        return X_Train, X_Test, y_train, y_test
    

    def mse(error):
        return (1/2)*np.mean(error**2)
    
    def gradient_calculation(y, mX, w):
        error = y - mX.dot(w)
        gradient = -mX.T.dot(error) / len(error)
        return gradient, error
    
    """Normalize data"""
    def normalize(X):
        return (X - X.mean(axis=0))/ X.std(axis=0)
    
    """Find Mx and add it to dataframe"""
    def add_intercept_LR(X, y):
        samples = y.shape[0]
        X["INTERCEPT"] = pd.Series(np.ones(samples))
        return X
    
    """Data Processing"""
    def preprocessing(X, y, r, s):
        X_train, X_test, y_train, y_test = linear_adagrad.split(X, y, r, randomness=s)
        X_train = linear_adagrad.normalize(X_train)
        X_train = linear_adagrad.add_intercept_LR(X_train, y)
        X_test = linear_adagrad.normalize(X_test)
        X_test = linear_adagrad.add_intercept_LR(X_test, y)
    
        return X_train, X_test, y_train, y_test
    
    #""" function to find square root """"
    def sqrt_func(array):
        n = len(array)
        for i in range(n):
            array[i] = math.sqrt(array[i])
        return array
    
#    """Function for AdaGradient """"
    def adagrad_gradient_descent(y, mX, intial_weights, epochs, alpha=0.001, epsilon = 10e-6, eta=0.01 ):
        ws = [intial_weights]
        cost = []
        w = intial_weights
        previous = math.inf
        r = 0.0
        deltaweight = 0.0
        delta = math.pow(10,-7)
        for i in range(epochs):
            gradient, error = linear_adagrad.gradient_calculation(y, mX, w)
            r = r + gradient*gradient
            loss = np.sqrt(2 * linear_adagrad.mse(error))
            # Update
            deltaweight = ((-alpha/(delta+linear_adagrad.sqrt_func(r)))*gradient)
            w = w + deltaweight
            
            ws.append(w)
            cost.append(loss)
            # convergence
            if(abs(loss - previous) < epsilon) :
                print("Converged")
                break
            previous = loss
        return cost, ws





if __name__ == "__main__":

    dataset = load_boston()
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(\
                    X, y, test_size=0.3, random_state=5)
    
    boston_dataset = load_boston()
    boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    # Creating a new column and Storing the target or y values
    boston['MEDV'] = boston_dataset.target
    
    # Assigning columns LSTAT and RM that has high correlation
    X = boston[['LSTAT', 'RM']]
    y = boston.MEDV
    
    X_train, X_test, y_train, y_test = train_test_split(\
                    X, y, test_size=0.3, random_state=5)
    
    
    print("\n" "--------------------Gradient Algorithm for Linear Model -------------------")
    print("")
    print("--------------------------------------------")
    print("Algorithm  Gradient Descent")
    print("--------------------------------------------")
    print("")
    custom_model = LinearRegression(X_train, y_train).gradient_decent()
    our_train_accuracy = custom_model.rmse_finder()
    our_test_accuracy = custom_model.rmse_finder(X_test, y_test)
    print(pd.DataFrame([[our_train_accuracy],[our_test_accuracy]],
                 ['Training:', 'Test:'],    
                 ['RMSE   ']))
    
    print("")
    print("--------------------------------------------")
    print("Algorithm  Stochastic Gradient Descent")
    print("--------------------------------------------")
    print("")
    custom_model = LinearRegression(X_train, y_train).stochastic_gradient_descent()
    our_train_accuracy = custom_model.rmse_finder()
    our_test_accuracy = custom_model.rmse_finder(X_test, y_test)
    print(pd.DataFrame([[our_train_accuracy],[our_test_accuracy]],
                 ['Training:', 'Test:'],    
                 ['RMSE   ']))
    
    print("")
    print("--------------------------------------------")
    print("Algorithm  SGD with Momentum")
    print("--------------------------------------------")
    print("")
    custom_model = LinearRegression(X_train, y_train).momentum_gradient_descent()
    our_train_accuracy = custom_model.rmse_finder()
    our_test_accuracy = custom_model.rmse_finder(X_test, y_test)
    print(pd.DataFrame([[our_train_accuracy],[our_test_accuracy]],
                 ['Training:', 'Test:'],    
                 ['RMSE   ']))
    
    print("")
    print("--------------------------------------------")
    print("Algorithm  SGD with Nesterov Momentum")
    print("--------------------------------------------")
    print("")
    custom_model = LinearRegression(X_train, y_train).nesterov_gradient_descent()
    our_train_accuracy = custom_model.rmse_finder()
    our_test_accuracy = custom_model.rmse_finder(X_test, y_test)
    print(pd.DataFrame([[our_train_accuracy],[our_test_accuracy]],
                 ['Training:', 'Test:'],    
                 ['RMSE   ']))
    
    gamma = 0.9
    epochs = 100
    intial_weights = np.array([0, 0, 0])
    X_train, X_test, y_train, y_test = linear_adagrad.preprocessing(X,y,0.3,1)
    gradient_cost, weights = linear_adagrad.adagrad_gradient_descent(y_train, X_train, intial_weights, epochs, gamma)
    
    params = weights[-1]
    
    y_pred = params[2] + params[0]*X_test.LSTAT + params[1]*X_test.RM
    
    print("")
    print("--------------------------------------------")
    print("Algorithm  AdaGrad Algorithm")
    print("--------------------------------------------")
    print("")
    rmse_train = gradient_cost[-1]
    print("Training:  {:.6f}".format(rmse_train))
    rmse_test = np.sqrt(2 * linear_adagrad.mse(y_test - y_pred))
    print("Test:      {:.6f}".format(rmse_test))
