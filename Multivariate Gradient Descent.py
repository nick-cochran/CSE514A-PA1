# 514A Data Mining PA1B
# Becky Shofner
# Student #513917

# Question 2: code for gradient descent of multivariate linear regression model
# initialize weight vector of ones for each variable
m = [1, 1, 1]
b = 1
# learning rate
alpha = 0.1

# sample data with multiple variables
# X = [3, 4, 5]
X_samples = [[3, 4, 4], [4, 2, 1], [10, 2, 5], [3, 4, 5], [11, 1, 1]]

# target value
# y = 4
y_targets = [3, 2, 8, 4, 5]

# single step in gradient descent
max_iter = 1

for iteration in range(max_iter):
    # initialize vector of m_gradients of size m
    m_gradient = [0] * len(m)
    b_gradient = 0

    # loop through variable and target value each sample
    for index in range(len(X_samples)):
        X = X_samples[index]
        y = y_targets[index]

        # calculate prediction error
        prediction = sum(m[s] * X[s] for s in range(len(m))) + b
        error = y - prediction

        # loop through all features in sample and calculate error and gradient for both m and b
        for s in range(len(m)):
            m_gradient[s] += -2 * X[s] * error

        b_gradient += -2 * error

    # update m and b with learning rate and gradients
    for s in range(len(m)):
        m[s] = m[s] - alpha * m_gradient[s] / len(X_samples)

    b = b - alpha * b_gradient / len(X_samples)

print(f"Updated parameters m: {m} and b: {b}")