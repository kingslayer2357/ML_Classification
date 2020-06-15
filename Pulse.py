# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 18:51:43 2020

@author: kingslayer
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset
dataset=pd.read_csv("D:\work\ML A to Z\Own\Classification\Pulse.csv")

#Matrix and vector
X=dataset.iloc[:,[1,2,4,5,6,7,]].values
y=dataset.iloc[:,3].values



#training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)




#Applying PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
var=pca.explained_variance_ratio_

#Classifier
from sklearn.svm import SVC
regressor=SVC(kernel="rbf")
regressor.fit(X_train,y_train)


#prediction
y_pred=regressor.predict(X_test)

#cm
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)




#Appling k fold
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=regressor,X=X_train,y=y_train,cv=10)
m=accuracies.mean()


#plot

plt.scatter(X_train,y_train)
plt.plot(X_train,regressor.predict(X_train))
plt.show()

plt.scatter(X_test,y_test)
plt.plot(X_test,regressor.predict(X_test))
plt.show()