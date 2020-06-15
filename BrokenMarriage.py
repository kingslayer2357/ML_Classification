# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:53:06 2020

@author: kingslayer
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset
dataset=pd.read_csv("D:\work\ML A to Z\Own\Classification\BrokenMarriage.csv")


#splitting into matrix of features and dependant vector
X=dataset.iloc[:,1:3].values
y=dataset.iloc[:,4].values



#Encoding categorical variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encoder=LabelEncoder()
X[:,1]=encoder.fit_transform(X[:,1])
encoder2=LabelEncoder()
y=encoder2.fit_transform(y)

#Splitting into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#CLASSIFIER
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)


#predicting result
y_pred=classifier.predict(X_test)
y_pred=(0.5<y_pred)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Appling k fold
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=5)
m=accuracies.mean()