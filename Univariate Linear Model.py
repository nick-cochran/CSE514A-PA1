# 514A Data Mining PA1B
# Becky Shofner
# Student #513917
from statistics import LinearRegression


# Question 1.1
# Model: good VE should be higher than 1 and a MSE should be as close to 0
# Cement
# One model: alpha = 0.00001 and iterations = 100
# Second model: alpha = 0.000005 and iterations = 10

# Blast Furnace Slag
# One model: alpha =


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# load excel file
df = pd.read_excel('Concrete_Data.xls', sheet_name='Sheet1')
print(df.columns)

# learning rate
alpha = 0.00000999

# steps in gradient descent
max_iter = 10000

X = df['Cement (component 1)(kg in a m^3 mixture)'].to_numpy()
# normalized variable
# X_norm = df['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'] / df['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'].abs().max()
y = df['Concrete compressive strength(MPa, megapascals) '].to_numpy()

# initialize parameters
m = 1
b = 1

for i in range(max_iter):
    m_gradient = 0
    b_gradient = 0

    # loop through all samples and calculate error and gradient for both m and b
    for s in range(len(X)):
        error = y[s] - (m * X[s] + b)
        m_gradient += -2 * X[s] * error
        # error = y[s] - (m * X_norm[s] + b)
        # m_gradient += -2 * X_norm[s] * error
        b_gradient += -2 * error

    # update m and b with learning rate and gradients
    m = m - alpha * m_gradient / len(X)
    b = b - alpha * b_gradient / len(X)
    # m = m - alpha * m_gradient / len(X_norm)
    # b = b - alpha * b_gradient / len(X_norm)

# evaluate model at end of gradient descent with variance explained
y_pred = m * X + b
mse = np.mean((y - y_pred) ** 2)
ve = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

print(f'Predictor: Cement, Updated parameters m: {m}, b: {b}, MSE: {mse}, and VE: {ve:.4f}')