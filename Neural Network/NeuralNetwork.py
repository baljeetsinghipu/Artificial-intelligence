from torch.utils.data import Dataset
import numpy as np # linear algebra
import pandas as pd
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn



class Titanic:
    
    def __init__(self):
        self.data = pd.read_csv('titanic.csv',comment='#')
        self.data_transformation(self.data)

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


obj = Titanic()

X_train, X_test, y_train, y_test = obj.split_data();

class MinstDataset(data.Dataset):
    def __init__(self):
        train = X_train.values
        train_labels = y_train.values
#        train = train.drop("Survived",axis=1).values
        self.datalist = train
        self.labels = train_labels
    def __getitem__(self, index):
        return torch.Tensor(self.datalist[index].astype(float)), self.labels[index]
    def __len__(self):
        return self.datalist.shape[0]
    
    
train_Set = MinstDataset()

trainloader = torch.utils.data.DataLoader( dataset = train_Set , batch_size= 10 , shuffle = True)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 2)
        
        self.dropout = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)
    
model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)


from torch.autograd import Variable
epochAmount = 500;
for epoch in range(epochAmount):
    for i, (data, labels) in enumerate(trainloader):
#        print("sdsd");
        data = Variable(data)
        labels = Variable(labels)
        
        optimizer.zero_grad()
        outputs = model(data)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
#        print(i)
        if (i+1) % 5 == 0:
            print ('Epoch [%d/%d], Iter [%d] Loss: %.4f' %(epoch+1, epochAmount, i+1, loss.data))
            

model.eval()
train = X_train
train = train.values.astype(float)
train = Variable(torch.Tensor(train))
pred = model(train)
_, predlabel = torch.max(pred.data, 1)
predlabel = predlabel.tolist()

print("")
print("-------------------Neural Network - Titanic Datset -------------------")
print("")
print("------------------------------------------------------------")
print("Report on Training Data")
print("------------------------------------------------------------")
print(classification_report(y_train,predlabel))
print("Accuracy:",accuracy_score(y_train, predlabel)*100,"%")

model.eval()
test = X_test
test = test.values.astype(float)
test = Variable(torch.Tensor(test))
pred = model(test)
_, predlabel = torch.max(pred.data, 1)
predlabel = predlabel.tolist()

print("")
print("------------------------------------------------------------")
print("Report on Testing Data")
print("------------------------------------------------------------")
print(classification_report(y_test,predlabel))
print("Accuracy:",accuracy_score(y_test, predlabel)*100,"%")

class LinearRegression:
    @staticmethod
    def linear_regression_using_py_torch(boston):
        X, y = (boston.data, boston.target)
        dim = X.shape[1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=8000)

        torch.set_default_dtype(torch.float64)
        
        net = nn.Sequential(
            nn.Linear(dim, 16, bias=True), nn.ReLU(),
            nn.Linear(16, 32, bias=True), nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        criterion = nn.MSELoss()
        opt = torch.optim.SGD(net.parameters(), lr=0.01)

        num_epochs = 20

        y_train_t = torch.from_numpy(y_train).clone().reshape(-1, 1)
        x_train_t = torch.from_numpy(X_train).clone()

        for i in range(num_epochs):
            y_hat = net(x_train_t)
            loss = criterion(y_train_t, y_hat)
            loss.backward()
            opt.step()
            opt.zero_grad()
            
        y_prediction = net(torch.from_numpy(X_test).detach())
        error = y_prediction.detach().numpy() - y_test
        print("RMSE of testing set:", np.sqrt(np.mean(error * error)))
        
        ##Please Uncomment for the visulalization:-
#        plt.plot(y_prediction.detach().numpy(), y_test, '+')
#        plt.show()

print("")
print("-------------------Neural Network Boston Dataset -------------------")
print("")
boston_dataset = load_boston()
LinearRegression.linear_regression_using_py_torch(boston_dataset)

    