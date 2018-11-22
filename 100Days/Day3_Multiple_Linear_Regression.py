#!-*-coding:utf-8 -*-
#!@Author:Chenguiyuan

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv('F:\Mycode\Data\\50_Startups.csv')
X = data.iloc[:, 0:4].values
Y = data.iloc[:, 4].values

labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=False, test_size=0.25)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)
print(regressor.score(X_test, Y_test))
