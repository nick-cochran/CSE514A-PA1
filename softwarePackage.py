# 514A Data Mining PA1 C
# Nick Cochran and Becky Shofner


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from patsy.highlevel import dmatrices, dmatrix
# import patsy. as pt

# Normalized features with alpha = 0.1 and iterations = 10000:
# Multivariate Predictors: Updated parameters m: [ 61.21979373  34.59009987  15.88069725 -45.0341261    8.49825487
#   11.78385602  11.24234976  41.6119085 ], b: 0.018549330490209133, MSE: 107.2916381020434, and VE: 0.6152

# Raw features with alpha = 0.0000000001 and iterations = 10000000
# Multivariate Predictors: Updated parameters m: [ 0.08860257  0.05621522  0.04656012  0.08461062  0.94619578 -0.0123551
#  -0.01273037  0.08832292], b: 0.9970172780469101, MSE: 137.50502519032116, VE: 0.5068

# load excel file
df = pd.read_excel('Concrete_Data.xls', sheet_name='Sheet1')

# create copy to normalize variables
# df_normalized = df.copy()
#
# for column in df_normalized.columns:
#     # normalize all variables but response variable
#     if column != 'Concrete compressive strength(MPa, megapascals) ':
#         df_normalized[column] = df_normalized[column] / df_normalized[column].abs().max()
#
# print(df_normalized.head())

# List of predictor variables
predictors = ['Cement (component 1)(kg in a m^3 mixture)',
              'Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
              'Fly Ash (component 3)(kg in a m^3 mixture)',
              'Water  (component 4)(kg in a m^3 mixture)',
              'Superplasticizer (component 5)(kg in a m^3 mixture)',
              'Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
              'Fine Aggregate (component 7)(kg in a m^3 mixture)',
              'Age (day)']
pres = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplasticizer', 'CoarseAggregate', 'FineAggregate', 'Age']
target_value = ['Concrete compressive strength(MPa, megapascals) ']

# X = df_normalized[predictors].to_numpy()
X = df[predictors].to_numpy()

# y = dmatrix(target_value, data=df, return_type='dataframe')
# X = dmatrix(predictors, data=df, return_type='dataframe')

# target values
# y= df_normalized['Concrete compressive strength(MPa, megapascals) '].to_numpy()
y = df['Concrete compressive strength(MPa, megapascals) '].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=900, test_size=130, random_state=0)


model = sm.OLS(y_train, X_train).fit()
print(model.summary())

print('\n\n\n Testing Data: \n\n\n')

model = sm.OLS(y_test, X_test).fit()
print(model.summary())

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse:.4f}')




# # code for gradient descent of multivariate linear regression model
# # initialize with ones - one weight for each feature
# m = np.ones(X.shape[1])
# b = 1
# # learning rate
# alpha = 0.1
#
# # single step in gradient descent
# max_iter = 10000
#
# # faster calculations using vector operations
# for iteration in range(max_iter):
#     y_pred = np.dot(X, m) + b
#
#     # Calculate errors
#     error = y - y_pred
#
#     # Calculate gradients for weights and bias
#     m_gradient = -2 * np.dot(X.T, error) / len(y)
#     b_gradient = -2 * np.sum(error) / len(y)
#
#     # Update weights and bias
#     m = m - alpha * m_gradient
#     b = b - alpha * b_gradient
#
#
# # evaluate model at end of gradient descent with variance explained
# # y_pred = m * X + b
# y_pred = np.dot(X, m) + b
# mse = np.mean((y - y_pred) ** 2)
# ve = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
#
# print(f'Updated parameters m: {m}, b: {b}, MSE: {mse}, VE: {ve:.4f}')


