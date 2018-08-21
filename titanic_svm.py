#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 21:53:05 2018

@author: amascaro_1
"""

#import re
import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
#from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV

testData = pd.read_csv("/Users/amascaro_1/kaggle/titanic/test.csv")
trainData = pd.read_csv("/Users/amascaro_1/kaggle/titanic/train.csv")

meanClass1 = trainData[trainData['Pclass'] == 1]['Age'].mean()
meanClass2 = trainData[trainData['Pclass'] == 2]['Age'].mean()
meanClass3 = trainData[trainData['Pclass'] == 3]['Age'].mean()

def impute_age(cols): 
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return meanClass1
        elif Pclass == 2:
            return meanClass2
        else:
            return meanClass3
    else:
        return Age
    
trainData['Age'] = trainData[['Age','Pclass']].apply(impute_age,axis=1)
testData['Age'] = testData[['Age','Pclass']].apply(impute_age,axis=1)

trainData.loc[:,'Cabin'] = trainData['Cabin'].isnull()

testData.loc[:,'Cabin'] = testData['Cabin'].isnull()
testData.loc[testData['Fare'].isnull(),'Fare'] = np.mean(testData['Fare'])

sex = pd.get_dummies(trainData['Sex'],drop_first=True)
sexTest = pd.get_dummies(testData['Sex'],drop_first=True)

embarked = pd.get_dummies(trainData['Embarked'],drop_first = True)
embarkedTest = pd.get_dummies(testData['Embarked'],drop_first = True)

trainData = pd.concat([trainData,sex,embarked],axis=1)
testData= pd.concat([testData,sexTest,embarkedTest],axis=1)


#Engineer some features:
for df in trainData, testData:
    df['Family Size'] = df['SibSp'] + df['Parch'] + 1

title = list()
for str in trainData['Name'].tolist(): 
    splitStr = str.split('.')
    title.append(splitStr[0].split(',')[1])
trainData = pd.concat([trainData,pd.DataFrame({'Title' :title})],axis=1)

title = list()
for str in testData['Name'].tolist(): 
    splitStr = str.split('.')
    title.append(splitStr[0].split(',')[1])
testData = pd.concat([testData,pd.DataFrame({'Title' :title})],axis=1)


titles = pd.get_dummies(trainData['Title'])
titles.rename(columns=lambda x: x.strip(),inplace=True)
rareTitleStr = list()
rareTitle = titles['Mlle']
for col in titles:
    if titles[col].sum() < 50:
        rareTitle = rareTitle | titles[col]
        rareTitleStr.append(titles[col].name)
rareTitle.rename('Rare Title',inplace=True)

titles.drop(rareTitleStr,axis=1,inplace=True)
trainData = pd.concat([trainData,rareTitle,titles],axis=1)

titles = pd.get_dummies(testData['Title'])
titles.rename(columns=lambda x: x.strip(),inplace=True)
rareTitleStr = list()
rareTitle = titles['Dr']
for col in titles:
    if titles[col].sum() < 50:
        rareTitle = rareTitle | titles[col]
        rareTitleStr.append(titles[col].name)
rareTitle.rename('Rare Title',inplace=True)

titles.drop(rareTitleStr,axis=1,inplace=True)
testData = pd.concat([testData,rareTitle,titles],axis=1)

trainData.drop(['Sex','Embarked','Ticket','PassengerId','Title','Name'],axis=1,inplace=True)
testData.drop(['Sex','Embarked','Ticket','Title','Name'],axis=1,inplace=True)


scaler = StandardScaler()
scaler.fit(trainData[['Fare','Age']])
scaled_data = scaler.transform(trainData[['Fare','Age']])
trainData = pd.concat([trainData.drop(['Fare','Age'],axis=1),pd.DataFrame(scaled_data,columns=['Fare','Age'])],axis=1)

scaler = StandardScaler()
scaler.fit(testData[['Fare','Age']])
scaled_data = scaler.transform(testData[['Fare','Age']])
testData = pd.concat([testData.drop(['Fare','Age'],axis=1),pd.DataFrame(scaled_data,columns=['Fare','Age'])],axis=1)


x = trainData.drop('Survived',axis=1)
y = trainData['Survived']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

#model = SVC()
#
#model.fit(x_train,y_train)
#svm_preds = model.predict(x_test)

param_grid = {'C':[0.1,1,10,100,1000,10000],'gamma':[1,0.5,0.1,0.06,0.05,0.03,0.04,0.01]}
grid = GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(x_train,y_train)

grid_preds = grid.predict(x_test)


#dtree = DecisionTreeClassifier()
#
#dtree.fit(x_train,y_train)
#
#preds = dtree.predict(x_test)
#
#print(confusion_matrix(y_test,preds))
#print(classification_report(y_test,preds))
#
## Now try a random forest instead of just a straight decision tree:
#from sklearn.ensemble import RandomForestClassifier
#
#rfc = RandomForestClassifier(n_estimators = 400)
#
#rfc.fit(x_train,y_train)
#rfc_preds = rfc.predict(x_test)

print(confusion_matrix(y_test,grid_preds))
print(classification_report(y_test,grid_preds))

grid.fit(x,y)
trueTestPredictions = grid.predict(testData.drop('PassengerId',axis=1))

trueTestOut = pd.concat([testData['PassengerId'],pd.DataFrame({'Survived': trueTestPredictions.T.tolist()})],axis=1,ignore_index=True)

trueTestOut.to_csv("/Users/amascaro_1/kaggle/titanic/svm_1.csv",index=False,header=False)


