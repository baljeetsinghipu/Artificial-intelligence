import numpy as np

import pandas as pd
import seaborn as sns
import seaborn as seabornInstance
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score



class LinearRegressionModels:
    
    def __init__(self):
        boston_dataset = load_boston()
        self.linearRegressionBoston(boston_dataset)
        
    def linearRegressionBoston(self,dataset):
        np.random.seed(1)
        boston = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        boston.head()
        boston['MEDV'] = dataset.target
        boston.isnull().sum()
        
        X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns=['LSTAT', 'RM'])
        Y = boston['MEDV']
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3, random_state=5)
        print(X_train.head())
        print(X_train.shape, Y_train.shape)
#        exit(0)
        self.model_creation(X_test, X_train, Y_test, Y_train)
        
        ''' # Please uncomment for the visualization
        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        sns.distplot(boston['MEDV'], bins=30)
        plt.show()
        correlation_matrix = boston.corr().round(2)
        sns.heatmap(data=correlation_matrix, annot=True)
        plt.figure(figsize=(20, 5))
        
        features = ['LSTAT', 'RM']
        target = boston['MEDV']
        for i, col in enumerate(features):
            plt.subplot(1, len(features), i + 1)
            x = boston[col]
            y = target
            plt.scatter(x, y, marker='o')
            plt.title(col)
            plt.xlabel(col)
            plt.ylabel('MEDV')
        '''
        

    def model_creation(self,X_test, X_train, Y_test, Y_train):
        lin_model = LinearRegression()
        lin_model.fit(X_train, Y_train)
        # model evaluation for training set
        y_train_predict = lin_model.predict(X_train)
        rootMeanSquare = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
        print('The root mean square error for training set: {}'.format(rootMeanSquare))
        y_test_predict = lin_model.predict(X_test)
        rootMeanSquare = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
        print('The root mean square error for testing set: {}'.format(rootMeanSquare))

    def data_transform_split(self,dataset):
        np.random.seed(2)
        advertisingDataframe = pd.read_csv(dataset)
        
        advertisingDataframe.isnull().sum() * 100 / advertisingDataframe.shape[0] # filling null value with mean.
        
        X_train, X_test, y_train, y_test = train_test_split(advertisingDataframe['TV'].values.reshape(-1,1), advertisingDataframe['Sales'].values.reshape(-1,1), train_size=0.7, test_size=0.3, random_state=5)
        self.model_creation(X_test, X_train, y_test, y_train)
        
        '''# Please uncomment for the visualization
        fig, axs = plt.subplots(3, figsize=(5, 5))
        print(advertisingDataframe.head())
        #There are no outliers in the data
        sns.boxplot(advertisingDataframe['TV'])
        plt.show()
        sns.boxplot(advertisingDataframe['Radio'])
        plt.show()
        sns.boxplot(advertisingDataframe['Newspaper'])
        plt.show()
        sns.boxplot(advertisingDataframe['Sales'])
        plt.show()
        seabornInstance.distplot(advertisingDataframe['Sales'])
        sns.pairplot(advertisingDataframe, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales',
                     kind='scatter')
        sns.heatmap(advertisingDataframe.corr(), annot=True)
        #Its clear that Tv is more related to sales
        plt.show()
        '''
       
        
        

if __name__ == "__main__":
    print("")
    print("--------------------Linear Regression - Boston Housing -----------------------------")
    print("")
    obj=LinearRegressionModels()
    print("\n" "--------------------Linear Regression - advertising-data ----------------------------")
    print("")
    obj.data_transform_split("Advertising.csv")

    

