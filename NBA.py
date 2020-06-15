# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:11:51 2020

@author: kingslayer
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset
dataset=pd.read_csv("D:\\work\\ML A to Z\\Own\\Classification\\NBA.csv")

#Matrix and vector
X=dataset.iloc[:,1:20].values
y=dataset.iloc[:,20].values

#missing Data
from sklearn.impute import SimpleImputer as Imputer
imputer=Imputer()
X[:,8:9]=imputer.fit_transform(X[:,8:9])

#training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


#ANN
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

classifier.add(Dense(output_dim=10,init="uniform",activation="relu",input_dim=19))
classifier.add(Dense(output_dim=10,init="uniform",activation="relu"))
classifier.add(Dense(output_dim=10,init="uniform",activation="relu"))
classifier.add(Dense(output_dim=10,init="uniform",activation="relu"))
classifier.add(Dense(output_dim=10,init="uniform",activation="relu"))
classifier.add(Dense(output_dim=1,init="uniform",activation="sigmoid"))

classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=1000)

#predict
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

#CM
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)




#plot
plt.scatter(X_train[:,0],X_train[:,1],c=y_train)
plt.show()