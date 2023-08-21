# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 13:50:13 2023

@author: CG-CS-PC16
"""

import pandas as pd
df = pd.read_csv("train.csv")

df.info()

# display first 4 records
print(df.head(4))

print(df.isnull().sum())

df = df.drop(["Cabin"], axis=1)
df.info()

cols = ["Name","Fare"]
df = df.drop(cols, axis=1)
df.info()

df2 = df["Age"].mean()
df["Age"].fillna(43, inplace=True)
print(df.isnull().sum())

df["Embarked"].fillna("NA", inplace=True)
print(df.isnull().sum())

df.dropna(inplace=True)
print(df.info())

print(df.nunique())
print(df["Pclass"].value_counts())

df.drop_duplicates(inplace=True)
df.info()

x = df.values
y = df["Survived"].values
x = df.drop("Survived", axis=1).values

# split the data into training and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split
print("x train shape: ", x_train.shape)
print("y train shape: ", y_train.shape)
print("x test shape: ", x_test.shape)
print("y test shape: ", y_test.shape)




