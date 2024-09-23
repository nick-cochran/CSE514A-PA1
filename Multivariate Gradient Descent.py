# 514A Data Mining PA1B
# Becky Shofner
# Student #513917
from typing import List

# Question 2
# code for gradient descent of multivariate linear regression model
# one weight for each feature
m = [1, 1, 1]
b = 1
# learning rate
alpha = 0.1

# sample data with features and targets
# multivariate input with 3 features
# X = [3, 4, 5]
X_samples = [
    [3, 4, 4],
    [4, 2, 1],
    [10, 2, 5],
    [3, 4, 5],
    [11, 1, 1]
]
# target value
# y = 4
y_targets = [3, 2, 8, 4, 5]

# single step in gradient descent
max_iter = 1

for iteration in range(max_iter):
    # account for all features and weights
    m_gradients = [0] * len(m)
    b_gradient = 0

    # loop through each sample
    for index in range(len(X_samples)):
        X = X_samples[index]
        y = y_targets[index]

        # calculate prediction error
        prediction = sum(m[j] * X[j] for j in range(len(m))) + b
        error = y - prediction

        # loop through all features in sample and calculate error and gradient for both m and b
        for j in range(len(m)):
            m_gradients[j] += -2 * X[j] * error

        b_gradient += -2 * error

    # update m and b with learning rate and gradients
    for j in range(len(m)):
        m[j] = m[j] - alpha * m_gradients[j] / len(X_samples)

    b = b - alpha * b_gradient / len(X_samples)

print(f"Updated parameters m: {m} and b: {b}")
