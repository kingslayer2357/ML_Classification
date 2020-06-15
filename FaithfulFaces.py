# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 02:08:44 2020

@author: kingslayer
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset=pd.read_csv("D:\work\ML A to Z\Own\Classification\FaithfulFaces.csv")

#X an y
X=dataset.iloc[:,[1,2,4,5,6,7]].values
y=dataset.iloc[:,3].values

#Encoding
from sklearn.preprocessing import LabelEncoder
encoder1=LabelEncoder()
X[:,4]=encoder1.fit_transform(X[:,4])
encoder2=LabelEncoder()
X[:,5]=encoder1.fit_transform(X[:,5])

#training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


#Applying ANN
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

classifier.add(Dense(output_dim=4,activation="relu",input_dim=6,init="uniform"))
classifier.add(Dense(output_dim=4,activation="relu",init="uniform"))
classifier.add(Dense(output_dim=4,activation="relu",init="uniform"))
classifier.add(Dense(output_dim=1,activation="sigmoid",init="uniform"))

classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=1000)

#predicting
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

#cm
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)



