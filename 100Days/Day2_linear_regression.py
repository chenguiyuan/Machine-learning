#!-*-coding:utf-8 -*-
#!@Author:Chenguiyuan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import normalize
from sklearn import linear_model
from sklearn.model_selection import train_test_split

data = pd.read_csv('F:\Mycode\Data\studentscores.csv')
X = data.iloc[:, 0].values.reshape(-1, 1)
Y = data.iloc[:, 1].values.reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
model = linear_model.LinearRegression()
model = model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

plt.scatter(X_train, Y_train, c='red')
plt.plot(X_train, model.predict(X_train), c='blue')

plt.scatter(X_test, Y_test, c='green')
plt.plot(X_test, model.predict(X_test), c='black')
plt.show()


