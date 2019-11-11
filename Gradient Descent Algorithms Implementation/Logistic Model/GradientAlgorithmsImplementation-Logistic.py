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


class LogisticRegressionGradientAlgorithm:
    def __init__(self, lr=0.01, number_of_iterations=100000, verbose=False):
        self.lr = lr
        self.number_of_iterations = number_of_iterations
        self.verbose = verbose
        self.theta = []

    @staticmethod
    def add_intercept_LR(X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    @staticmethod
    def sigmoid(z):
        higher_precision_z = np.array(z, dtype=np.float128)
        return 1 / (1 + np.exp(-higher_precision_z))

    @staticmethod
    def loss(h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    #gradient descent 
    def gradient_descent_LR_fit(self, X, y):

        self.theta = np.zeros(X.shape[1])
        for i in range(self.number_of_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            if self.verbose and i % 10000 == 0:
                z = np.dot(X, self.theta)
                h = self.sigmoid(z)
                print(f'loss: {self.loss(h, y)} \t')
    #Stochastic gradient descent finder
    def Stochastic_gradient_descent_LR_fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        for i in range(self.number_of_iterations):
            for point_x, point_y in zip(X, y):
                z = np.dot(point_x, self.theta)
                h = self.sigmoid(z)
                gradient = np.dot(point_x.T, (h - point_y))
                self.theta -= (self.lr * gradient)
    #SGT with Momentum
    def Stochastic_Momentum_gradient_descent_LR_fit(self, X, y, mu, nesterov=False):
        velocity = np.zeros(X.shape[1])
        self.theta = np.zeros(X.shape[1])
        for i in range(self.number_of_iterations):
            for point_x, point_y in zip(X, y):
                z = np.dot(point_x, self.theta)
                h = self.sigmoid(z)
                gradient = np.dot(point_x.T, (h - point_y))
                if nesterov:
                    self.theta += mu * velocity
                velocity = mu*velocity + self.lr * gradient
                self.theta -= velocity
    # SGT Nesterov_gradient_descent
    def Stochastic_Nesterov_gradient_descent_LR_fit(self, X, y, mu):
        self.Stochastic_Momentum_gradient_descent_LR_fit(X, y, mu, nesterov=True)

    # Adagrad Gradient
    def Stochastic_Adagrad_gradient_descent_LR_fit(self, X, y):
        cache = np.zeros(X.shape[1])
        self.theta = np.zeros(X.shape[1])
        for i in range(self.number_of_iterations):
            for point_x, point_y in zip(X, y):
                z = np.dot(point_x, self.theta)
                h = self.sigmoid(z)
                gradient = np.dot(point_x.T, (h - point_y))
                cache += gradient ** 2
                self.theta -= (self.lr * gradient) / (np.sqrt(cache) + 1e-8)


    def predict_prob(self, x):
        return self.sigmoid(np.dot(x, self.theta))

    def predict(self, x, threshold):
        return self.predict_prob(x) >= threshold
    

class LogisticRegressionMain:

    def __init__(self, dataset):
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessing_data(dataset)
        self.X_train = MinMaxScaler().fit_transform(self.X_train)
        self.X_test = MinMaxScaler().fit_transform(self.X_test)

    @staticmethod
    def preprocessing_data(dataset):
        dataset['Age'] = dataset[['Age', 'Pclass']].apply(LogisticRegressionMain.impute_age, axis=1)

        dataset.drop('Cabin', axis=1, inplace=True)
        dataset.dropna(inplace=True)
        sex = pd.get_dummies(dataset['Sex'], drop_first=True)
        embark = pd.get_dummies(dataset['Embarked'], drop_first=True)

        dataset.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
        dataset = pd.concat([dataset, sex, embark], axis=1)


        return train_test_split(dataset.drop('Survived', axis=1), dataset['Survived'],
                                test_size=0.20, random_state=101)

    def logistic_regression_titanic(self):
        
        LogisticRegressionMain.start_gradient_Descent(self.X_train, self.y_train,self.X_test, self.y_test)
    @staticmethod
    def impute_age(cols):
        Age = cols[0]
        Pclass = cols[1]
        if pd.isnull(Age):
            if Pclass == 1:
                return 37
            elif Pclass == 2:
                return 29
            else:
                return 24
        else:
            return Age

   
    def start_gradient_Descent(X_train, y_train, X_test, y_test):

        model1 = LogisticRegressionGradientAlgorithm(lr=0.1, number_of_iterations=200)
        model1.gradient_descent_LR_fit(X_train, y_train)
        LogisticRegressionMain.print_statistics_report(X_test, X_train, model1, y_test, y_train, "Gradient Descent")
        
        model2 = LogisticRegressionGradientAlgorithm(lr=0.1, number_of_iterations=100)
        model2.Stochastic_gradient_descent_LR_fit(X_train, y_train)
        LogisticRegressionMain.print_statistics_report(X_test, X_train, model2, y_test, y_train,
                                                   "Stochastic Gradient Descent")
        
        model3 = LogisticRegressionGradientAlgorithm(lr=0.1, number_of_iterations=100)
        model3.Stochastic_Momentum_gradient_descent_LR_fit(X_train, y_train, 0.9)
        LogisticRegressionMain.print_statistics_report(X_test, X_train, model3, y_test, y_train, "Stochastic Gradient Descent with Momentum")

        model4 = LogisticRegressionGradientAlgorithm(lr=0.1, number_of_iterations=100)
        model4.Stochastic_Nesterov_gradient_descent_LR_fit(X_train, y_train, 0.9)
        LogisticRegressionMain.print_statistics_report(X_test, X_train, model4, y_test, y_train,
                                                   "Stochastic Gradient Descent with Nesterov Momentum")

        model5 = LogisticRegressionGradientAlgorithm(lr=0.1, number_of_iterations=100)
        model5.Stochastic_Adagrad_gradient_descent_LR_fit(X_train, y_train)
        LogisticRegressionMain.print_statistics_report(X_test, X_train, model5, y_test, y_train, "AdaGrad Algorithm")


    
    def print_statistics_report(X_test, X_train, model, y_test, y_train, algorithm_used):
        predictions = model.predict(X_train, 0.5)
        print("")
        print("-----------------------------------------------------")
        print("Algorithm ", algorithm_used)
        print("-----------------------------------------------------")
        print("")
        print("Training sets Report :")
        print("----------------------")
        print(classification_report(y_train, predictions))
        print("Accuracy:{:.2f}".format(accuracy_score(y_train, predictions) * 100))
        predictions = model.predict(X_test, 0.5)
        
        print("\n" "Testing sets Report:")
        print("-------------------------")
        print(classification_report(y_test, predictions))
        print("Accuracy:{:.2f}".format(accuracy_score(y_test, predictions) * 100))



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
    

    print("\n" "--------------------Gradient Algorithm for Logistic Regression  -------------------")
    titanic = pd.read_csv('titanic.csv')
    dataset_titanic = LogisticRegressionMain(titanic)
    dataset_titanic.logistic_regression_titanic()