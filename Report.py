import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray

# load excel file
df = pd.read_excel('Concrete_Data.xls', sheet_name='Sheet1')

# create copy to normalize variables
df_normalized = df.copy()

# List of predictor variables
predictors = ['Cement (component 1)(kg in a m^3 mixture)',
              'Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
              'Fly Ash (component 3)(kg in a m^3 mixture)',
              'Water  (component 4)(kg in a m^3 mixture)',
              'Superplasticizer (component 5)(kg in a m^3 mixture)',
              'Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
              'Fine Aggregate (component 7)(kg in a m^3 mixture)',
              'Age (day)']

features = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']
X = df_normalized[predictors].to_numpy()

# Response variable
# y = df['Concrete compressive strength(MPa, megapascals) '].to_numpy()
y = df_normalized['Concrete compressive strength(MPa, megapascals) '].to_numpy()

# Set different learning rates and max_iter for each predictor
alphas = [0.00001, 0.00007, 0.0001, 0.000001, 0.01, 0.000001, 0.000001, 0.0001]  # Example alphas
max_iters = [10000, 100000, 1000, 10000000, 1000, 10000000, 1000000, 100000]  # Example max iterations

# Set up subplots
fig, axes = plt.subplots(4, 2, figsize=(14, 12))
axes = axes.flatten()  # Flatten the 4x2 array into a 1D array for easier access

# Loop over each predictor
for idx, predictor in enumerate(predictors):
    X: ndarray = df[predictor].to_numpy()

    # Normalize the predictor (Min-Max normalization)
    X_norm = (X - X.min()) / (X.max() - X.min())

    # Initialize parameters
    m = 1
    b = 1

    # Use the specific alpha and max_iter for this predictor
    alpha = alphas[idx]
    max_iter = max_iters[idx]

    # Perform vectorized gradient descent
    for i in range(max_iter):
        y_pred = m * X_norm + b
        error = y - y_pred
        m_gradient = -2 * np.dot(X_norm, error) / len(X_norm)
        b_gradient = -2 * np.mean(error)

        # Update m and b
        m = m - alpha * m_gradient
        b = b - alpha * b_gradient

    # Calculate final predictions
    y_pred = m * X_norm + b
    mse = np.mean((y - y_pred) ** 2)
    ve = 1 - (mse / (np.sum((y - np.mean(y)) ** 2) / len(y)))

    # Plot each predictor in the corresponding subplot
    axes[idx].scatter(X_norm, y, color='green', label='True data')
    axes[idx].plot(X_norm, y_pred, color='red', label='Fitted line')
    axes[idx].set_xlabel(predictor)
    axes[idx].set_ylabel('Concrete compressive strength (MPa)')
    axes[idx].legend()
    axes[idx].set_title(f'Normalized {features[idx]} Data vs Concrete Compressive Strength')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.suptitle('Normalized Predictors Data vs Concrete Compressive Strength', fontsize=16, y=1.02)
plt.show()
