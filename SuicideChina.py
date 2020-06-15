# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:43:09 2020# -*- coding: utf-8 -*-
"""



#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset
dataset=pd.read_csv("D:\work\ML A to Z\Own\Classification\SuicideChina.csv")

#Matrix and vector
X_see=dataset.iloc[:,[2,4,5,6,7,8,9,10,11]]
X=dataset.iloc[:,[2,4,5,6,7,8,9,10,11]].values
y=dataset.iloc[:,3].values

#Encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encoder0=LabelEncoder()
X[:,0]=encoder0.fit_transform(X[:,0])
encoder1=LabelEncoder()
X[:,1]=encoder1.fit_transform(X[:,1])
encoder4=LabelEncoder()
X[:,4]=encoder4.fit_transform(X[:,4])
encoder6=LabelEncoder()
X[:,6]=encoder6.fit_transform(X[:,6])
encoder7=LabelEncoder()
X[:,7]=encoder7.fit_transform(X[:,7])
encoder8=LabelEncoder()
X[:,8]=encoder8.fit_transform(X[:,8])
h_encoder=OneHotEncoder(categorical_features=[6])
X=h_encoder.fit_transform(X).toarray()


#training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)





#Classifier
from sklearn.linear_model import LogisticRegression as LR
regressor=LR(max_iter=1000,solver="saga",C=10)
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

#Grid Search
from sklearn.model_selection import GridSearchCV
param=[{"C":[1,10,100]}]
grid=GridSearchCV(estimator=regressor,param_grid=param,scoring="accuracy",cv=10,n_jobs=-1)
grid=grid.fit(X_train,y_train)
best=grid.best_params_






