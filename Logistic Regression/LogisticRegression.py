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




class Titanic:
    
    def __init__(self):
        self.data = pd.read_csv('titanic.csv',comment='#')
        self.data_transformation(self.data)
        X_train, X_test, y_train, y_test=self.split_data()
        logmodel = self.model(X_train,y_train);
        self.evaluate_model(X_test,y_test,X_train,y_train,logmodel)

    def data_transformation(self,data):
        self.data['Age'] = self.data[['Age','Pclass']].apply(self.data_transformation_impute_age,axis=1)
        self.dropping_irrelevant_columns()
        
    def data_transformation_impute_age(self,cols):
        Age = cols[0]
        Pclass = cols[1]
        #Replace null value with average age
        if pd.isnull(Age):
            if Pclass == 1:
                return 37 
            elif Pclass == 2:
                return 29
            else:
                return 24
        else:
            return Age
    
    def dropping_irrelevant_columns(self):
        self.data.drop('Cabin',axis=1,inplace=True)
#        test.drop('Cabin',axis=1,inplace=True)
        self.data.dropna(inplace=True)
        sex = pd.get_dummies(self.data['Sex'],drop_first=True)
        embark = pd.get_dummies(self.data['Embarked'],drop_first=True)

        self.data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
        self.data = pd.concat([self.data,sex,embark],axis=1)


    def split_data(self):
        # Splitting data  sets into training and testing sets 
        X_train, X_test, y_train, y_test = train_test_split(self.data.drop('Survived',axis=1),self.data['Survived'], test_size=0.20,random_state=101)
        
        return X_train, X_test, y_train, y_test
        
    def model(self,X_train,y_train):
        
        # Select Logistic Regression model
        logmodel = LogisticRegression(solver='lbfgs',max_iter=5000)
        # Training our model
        logmodel.fit(X_train,y_train)
        
        return logmodel
        
    def evaluate_model(self,X_test,y_test,X_train,y_train,logmodel):
        #Predictions on training set data 
        
        predictions = logmodel.predict(X_test)
        print("")
        print("-------------------Logistic Regression - Boston-----------------")
        print("")
        print("------------------------------------------------------------")
        print("Report on Testing Data")
        print("------------------------------------------------------------")
        
        print(classification_report(y_test,predictions))
        print("Accuracy:",accuracy_score(y_test, predictions)*100,"%")

        print("")
        print("------------------------------------------------------------")
        print("Report on Training Data")
        print("------------------------------------------------------------")
        predictions = logmodel.predict(X_train)
#
        print(classification_report(y_train,predictions))
        print("Accuracy:",accuracy_score(y_train, predictions)*100,"%")
        




class Prima_Indians:
    
    """
    Dataset
    -------
    
    The dataset includes data from 768 women with 8 characteristics, in particular:
    
    Number of times pregnant : NumTimesPregnant
    Plasma glucose concentration a 2 hours in an oral glucose tolerance test : PlasmaGlucose
    Diastolic blood pressure (mm Hg) : BloodPressure
    Triceps skin fold thickness (mm) : SkinThickness
    2-Hour serum insulin (mu U/ml) : TwoHourSerum
    Body mass index (weight in kg/(height in m)^2) : BMI
    Diabetes pedigree function : DiabetesFunction
    Age (years): Age
    The last column of the dataset indicates if the person has been diagnosed with diabetes (1) or not (0): HasDiabetes
    """
    
    def __init__(self):
        self.data = pd.read_csv('prima-indian.csv',comment='#')
        self.data.columns = ["NumTimesPregnant", "PlasmaGlucose", "BloodPressure","SkinThickness", "TwoHourSerum", "BMI","DiabetesFunction ", "Age", "HasDiabetes"]
        self.data_transformation()
        self.X_train, self.X_test, self.y_train, self.y_test=self.split_data()
        logmodel = self.model();
        self.evaluate_model(self.X_test,self.y_test,self.X_train,self.y_train,logmodel)
    
    def data_transformation(self):
        # Adding headers to the Data
        median_bmi = self.data['BMI'].median()
        # Substitute it in the BMI column of the
        # dataset where values are 0
        self.data['BMI'] = self.data['BMI'].replace(to_replace=0, value=median_bmi)

        # Calculate the median value for BloodPressure
        median_bloodp = self.data['BloodPressure'].median()
        # Substitute it in the BloodPressure column of the
        # dataset where values are 0
        self.data['BloodPressure'] = self.data['BloodPressure'].replace(to_replace=0, value=median_bloodp)

        # Calculate the median value for PlasmaGlucose
        median_plglcconc = self.data['PlasmaGlucose'].median()
        # Substitute it in the PlasmaGlucose column of the
        # dataset where values are 0
        self.data['PlasmaGlucose'] = self.data['PlasmaGlucose'].replace(to_replace=0, value=median_plglcconc)

        # Calculate the median value for TwoHourSerum
        median_twohourserins = self.data['TwoHourSerum'].median()
        # Substitute it in the TwoHourSerum column of the
        # dataset where values are 0
        self.data['TwoHourSerum'] = self.data['TwoHourSerum'].replace(to_replace=0, value=median_twohourserins)

    def split_data(self):
        # Splitting training sets 
        X_train, X_test, y_train, y_test = train_test_split(self.data.drop('HasDiabetes',axis=1),self.data['HasDiabetes'], test_size=0.20,random_state=101)
        return X_train, X_test, y_train, y_test
    
    def model(self):
        
        # Select Logistic Regression model
        logmodel = LogisticRegression(solver='lbfgs',max_iter=5000)
        # Training our model
        logmodel.fit(self.X_train,self.y_train)
        
        return logmodel
        
    def evaluate_model(self,X_test,y_test,X_train,y_train,logmodel):
        #Predictions on training set data 
        
        predictions = logmodel.predict(X_test)
       
        print("")
        print("-------------------Logisitic Regression - Advertising---------------")
        print("")
        print("------------------------------------------------------------")
        print("Report on Testing Data")
        print("------------------------------------------------------------")
        
        print(classification_report(y_test,predictions))
        print("Accuracy:",accuracy_score(y_test, predictions)*100,"%")

        print("")
        print("------------------------------------------------------------")
        print("Report on Training Data")
        print("------------------------------------------------------------")
        predictions = logmodel.predict(X_train)
#
        print(classification_report(y_train,predictions))
        print("Accuracy:",accuracy_score(y_train, predictions)*100,"%")
        
        

if __name__ == "__main__":

    Titanic()
    Prima_Indians()
    
    

