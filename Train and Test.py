# 514A Data Mining PA1B
# Becky Shofner
# Student #513917
# data from statistics import LinearRegression
# Geeks for Geeks: https://www.geeksforgeeks.org/how-to-do-train-test-split-using-sklearn-in-python/
# Sckit learn: https://scikit-learn.org/stable/modules/sgd.html#regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# load excel file
df = pd.read_excel('Concrete_Data.xls', sheet_name='Sheet1')

X = df['Cement (component 1)(kg in a m^3 mixture)'].to_numpy()
y = df['Concrete compressive strength(MPa, megapascals) '].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=900, test_size=130, random_state=41)

m = 0.11754320981681309
b = 1.2954941048512092


# Function to calculate predictions
def predict(X, m, b):
    return m * X + b


# Get predictions for training and test sets
y_train_pred = predict(X_train, m, b)
y_test_pred = predict(X_test, m, b)

# Training Data Plot
plt.scatter(y_train, y_train_pred, color='blue', label='Training data')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--', label='Fit')
plt.xlabel('Actual Compressive Strength')
plt.ylabel('Predicted Compressive Strength')
plt.title('Univariate (Concrete Raw) Training Data: Actual vs Predicted')
plt.legend()
plt.show()

# Testing Data Plot
plt.scatter(y_test, y_test_pred, color='green', label='Testing data')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Fit')
plt.xlabel('Actual Compressive Strength')
plt.ylabel('Predicted Compressive Strength')
plt.title('Univariate (Concrete Raw) Testing Data: Actual vs Predicted')
plt.legend()
plt.show()

mse = np.mean((y_test - y_test_pred) ** 2)
ve = 1 - (mse / (np.sum((y_test - np.mean(y_test)) ** 2) / len(y_test)))
print(f'Model performance: MSE: {mse}, VE: {ve:.4f}')
