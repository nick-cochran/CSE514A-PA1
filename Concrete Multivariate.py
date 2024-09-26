# 514A Data Mining PA1B
# Becky Shofner
# Student #513917

import pandas as pd
import numpy as np

# Normalized features with alpha = 0.1 and iterations = 10000:
# Multivariate Predictors: Updated parameters m: [ 61.21979373  34.59009987  15.88069725 -45.0341261    8.49825487
#   11.78385602  11.24234976  41.6119085 ], b: 0.018549330490209133, MSE: 107.2916381020434, and VE: 0.6152

# Raw features with alpha = 0.0000000001 and iterations = 10000000
# Multivariate Predictors: Updated parameters m: [ 0.08860257  0.05621522  0.04656012  0.08461062  0.94619578 -0.0123551
#  -0.01273037  0.08832292], b: 0.9970172780469101, MSE: 137.50502519032116, VE: 0.5068

# load excel file
df = pd.read_excel('Concrete_Data.xls', sheet_name='Sheet1')

# create copy to normalize variables
df_normalized = df.copy()

for column in df_normalized.columns:
    # normalize all variables but response variable
    if column != 'Concrete compressive strength(MPa, megapascals) ':
        df_normalized[column] = df_normalized[column] / df_normalized[column].abs().max()

print(df_normalized.head())

# Question 2.3: code for gradient descent of multivariate linear regression model
# List of predictor variables
predictors = ['Cement (component 1)(kg in a m^3 mixture)',
              'Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
              'Fly Ash (component 3)(kg in a m^3 mixture)',
              'Water  (component 4)(kg in a m^3 mixture)',
              'Superplasticizer (component 5)(kg in a m^3 mixture)',
              'Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
              'Fine Aggregate (component 7)(kg in a m^3 mixture)',
              'Age (day)']

X = df_normalized[predictors].to_numpy()
# X = df[predictors].to_numpy()

# target values
y = df['Concrete compressive strength(MPa, megapascals) '].to_numpy()

# code for gradient descent of multivariate linear regression model
# initialize with ones - one weight for each feature
m = np.ones(X.shape[1])
b = 1
# learning rate
alpha = 0.1

# single step in gradient descent
max_iter = 10000

for iteration in range(max_iter):
    # account for all features and weights and set to 0
    m_gradients = np.zeros(len(m))
    b_gradient = 0

    # loop through each sample
    for index in range(len(X)):
        x_sample = X[index]
        y_sample = y[index]

        # prediction = sum(m[j] * X[j] for j in range(len(m))) + b
        # calculate prediction error with dot product
        prediction = np.dot(m, x_sample) + b
        error = y_sample - prediction

        # loop through all features in sample and calculate error and gradient for both m and b
        for j in range(len(m)):
            m_gradients[j] += -2 * x_sample[j] * error
        b_gradient += -2 * error

    # update m and b with learning rate and gradients
    for j in range(len(m)):
        m[j] = m[j] - alpha * m_gradients[j] / len(X)

    b = b - alpha * b_gradient / len(X)

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


# evaluate model at end of gradient descent with variance explained
# y_pred = m * X + b
y_pred = np.dot(X, m) + b
mse = np.mean((y - y_pred) ** 2)
ve = 1 - mse / (np.sum((y - np.mean(y)) ** 2) / len(y))

print(f'Updated parameters m: {m}, b: {b}, MSE: {mse}, VE: {ve:.4f}')


