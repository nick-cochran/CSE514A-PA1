# Becky Shofner
# CSE514A Data Mining Programming Assignment 1B
# Geeks for Geeks: https://www.geeksforgeeks.org/multiple-linear-regression-with-scikit-learn/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# load excel file
df = pd.read_excel('Concrete_Data.xls', sheet_name='Sheet1')

# create copy to normalize variables
df_normalized = df.copy()

for column in df_normalized.columns:
    # normalize all variables but response variable
    if column != 'Concrete compressive strength(MPa, megapascals) ':
        df_normalized[column] = df_normalized[column] / df_normalized[column].abs().max()

print(df_normalized.head())

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
y = df_normalized['Concrete compressive strength(MPa, megapascals) '].to_numpy()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=130, random_state=42)

# Your learned weights from gradient descent
m = np.array([61.21979373, 34.59009987, 15.88069725, -45.0341261, 8.49825487, 11.78385602, 11.24234976, 41.6119085])
b = 0.018549330490209133


# Function to calculate predictions
def predict(X, m, b):
    return np.dot(X, m) + b


# Get predictions for training and test sets
y_train_pred = predict(X_train, m, b)
y_test_pred = predict(X_test, m, b)

# Training Data Plot
plt.scatter(y_train, y_train_pred, color='blue', label='Training data')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--', label='Fit')
plt.xlabel('Actual Compressive Strength')
plt.ylabel('Predicted Compressive Strength')
plt.title('Training Data: Actual vs Predicted')
plt.legend()
plt.show()

# Testing Data Plot
plt.scatter(y_test, y_test_pred, color='green', label='Testing data')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Fit')
plt.xlabel('Actual Compressive Strength')
plt.ylabel('Predicted Compressive Strength')
plt.title('Testing Data: Actual vs Predicted')
plt.legend()
plt.show()

mse = np.mean((y_test - y_test_pred) ** 2)
ve = 1 - np.sum((y_test - y_test_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
print(f'Model performance: MSE: {mse}, VE: {ve:.4f}')
