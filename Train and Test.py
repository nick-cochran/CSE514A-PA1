# 514A Data Mining PA1B
# Becky Shofner
# Student #513917
#data from statistics import LinearRegression
# Geeks for Geeks: https://www.geeksforgeeks.org/how-to-do-train-test-split-using-sklearn-in-python/
# Sckit learn: https://scikit-learn.org/stable/modules/sgd.html#regression

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# load excel file
df = pd.read_excel('Concrete_Data.xls', sheet_name='Sheet1')
print(df.columns)


X = df['Cement (component 1)(kg in a m^3 mixture)'].to_numpy()
# normalized variable
# X_norm = df['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'] / df['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'].abs().max()
y = df['Concrete compressive strength(MPa, megapascals) '].to_numpy()


X_train, X_test, y_train, y_test = train_test_split(train_size=0.8, test_size=0.2, random_state=0)

print('X_train : ')
print(X_train.head())
print(X_train.shape)
print('')
print('X_test : ')
print(X_test.head())
print(X_test.shape)
print('')

print('y_train : ')
print(y_train.head())
print(y_train.shape)
print('')
print('y_test : ')
print(y_test.head())
print(y_test.shape)

# model = LinearRegression()

