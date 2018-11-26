#!-*-coding:utf-8 -*-
#!@Author:Chenguiyuan

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
data = pd.read_csv('F:\Mycode\Data\Social_Network_Ads.csv')
X = data.iloc[:, 0:4].values
Y = data.iloc[:, 4].values
label_encoder = LabelEncoder()
X[:, 1] = label_encoder.fit_transform(X[:, 1])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=False, test_size=0.25)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
