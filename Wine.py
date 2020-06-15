# -*- coding: utf-8 -*-
"""
Created on Fri May  1 23:38:44 2020

@author: kingslayer
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset
dataset=pd.read_csv("D:\work\ML A to Z\Own\Classification\Wine.csv",sep=";")

#Matrix and vector
X=dataset.iloc[:,0:11].values
y=dataset.iloc[:,11].values

#Categorical Variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encoder=LabelEncoder()
y=encoder.fit_transform(y)
y=np.reshape(y,(-1,1))
onehotencoder=OneHotEncoder()
y=onehotencoder.fit_transform(y).toarray()

#training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)



#Applying PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=3)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
var=pca.explained_variance_ratio_

#ANN
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

classifier.add(Dense(output_dim=7,init="uniform",activation="relu",input_dim=6))
classifier.add(Dense(output_dim=7,init="uniform",activation="relu"))
classifier.add(Dense(output_dim=7,init="uniform",activation="relu"))
classifier.add(Dense(output_dim=7,init="uniform",activation="relu"))
classifier.add(Dense(output_dim=7,init="uniform",activation="relu"))
classifier.add(Dense(output_dim=7,init="uniform",activation="softmax"))

classifier.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
classifier.fit(X_train,y_train,nb_epoch=1000,batch_size=10)

#prediction
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)


#CM
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)




