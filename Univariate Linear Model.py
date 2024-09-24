# 514A Data Mining PA1B
# Becky Shofner
# Student #513917
from statistics import LinearRegression


# Question 1.1
# Model: good VE should be higher than 1 and a MSE should be as close to 0
# Cement Norm: alpha = 0.01, iterations = 100
# Cement Raw: alpha = 0.00001, iterations = 10000

# Slag Raw: alpha = 0.00007, iterations = 100000
# Slag Norm: alpha = 0.00007, iterations = 100000

# Fly Ash Norm: alpha = 0.1, iterations = 100
# Fly Ash Raw: alpha = 0.0001, iterations = 1000 (No positive positive found)

# Water Norm: alpha = 0.001, iterations = 100000
# Water Raw: alpha = 0.00001, iterations = 10000000

# Superplasticizer Norm: alpha = 0.01, iterations = 1000
# Superplasticizer Raw: alpha = 0.01, iterations = 1000

# Coarse Aggregate Norm: alpha = 0.1, iterations = 10000
# Coarse Aggregate Raw: alpha = 0.000001, iterations = 10000000 (No positive VE found)

# Fine Aggregate Norm: alpha = 0.1, iterations = 1000
# Fine Aggregate Raw: alpha = 0.000001, iterations = 10000000 (No positive VE found)

# Age Norm: alpha = 0.1, iterations = 100
# Age Raw: alpha = 0.0001, iterations = 100000


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray

# load excel file
df = pd.read_excel('Concrete_Data.xls', sheet_name='Sheet1')
print(df.columns)

# learning rate
alpha =  0.00001

# steps in gradient descent
max_iter = 10000

X: ndarray = df['Cement (component 1)(kg in a m^3 mixture)'].to_numpy()
# normalized variable
# X_norm = df['Cement (component 1)(kg in a m^3 mixture)'] / df['Cement (component 1)(kg in a m^3 mixture)'].abs().max()
y = df['Concrete compressive strength(MPa, megapascals) '].to_numpy()

# initialize parameters
m = 1
b = 1

# for i in range(max_iter):
#     m_gradient = 0
#     b_gradient = 0
#
#     # loop through all samples and calculate error and gradient for both m and b
#     for s in range(len(X)):
#         error = y[s] - (m * X[s] + b)
#         m_gradient += -2 * X[s] * error
#         # error = y[s] - (m * X_norm[s] + b)
#         # m_gradient += -2 * X_norm[s] * error
#         b_gradient += -2 * error
#
#     # update m and b with learning rate and gradients
#     m = m - alpha * m_gradient / len(X)
#     b = b - alpha * b_gradient / len(X)
#     # m = m - alpha * m_gradient / len(X_norm)
#     # b = b - alpha * b_gradient / len(X_norm)

# Vectorized gradient descent
for i in range(max_iter):
    # Calculate predictions
    y_pred = m * X + b

    # Calculate the error
    error = y - y_pred

    # Calculate gradients (vectorized)
    m_gradient = -2 * np.dot(X, error) / len(X)
    # m_gradient = -2 * np.dot(X_norm, error) / len(X_norm)
    b_gradient = -2 * np.mean(error)

    # Update m and b
    m = m - alpha * m_gradient
    b = b - alpha * b_gradient

# evaluate model at end of gradient descent with variance explained
y_pred = m * X + b
mse = np.mean((y - y_pred) ** 2)
ve = 1 - (mse / (np.sum((y - np.mean(y)) ** 2) / len(y)))

print(f'Predictor: Updated parameters m: {m}, b: {b}, MSE: {mse}, and VE: {ve:.4f}')

# Plot the results
plt.scatter(X, y, color='blue', label='True data')
plt.plot(X, y_pred, color='red', label='Fitted line')
plt.xlabel('Cement (component 1)(kg in a m^3 mixture)')
plt.ylabel('Concrete compressive strength (MPa)')
plt.legend()
plt.title('Raw Concrete Data vs Concrete Compressive Strength')
plt.show()