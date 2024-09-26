# 514A Data Mining PA1 C
# Nick Cochran and Becky Shofner

# DataRobot: https://www.datarobot.com/blog/multiple-regression-using-statsmodels/

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Normalized features with alpha = 0.1 and iterations = 10000:
# Multivariate Predictors: Updated parameters m: [ 61.21979373  34.59009987  15.88069725 -45.0341261    8.49825487
#   11.78385602  11.24234976  41.6119085 ], b: 0.018549330490209133, MSE: 107.2916381020434, and VE: 0.6152

# Raw features with alpha = 0.0000000001 and iterations = 10000000
# Multivariate Predictors: Updated parameters m: [ 0.08860257  0.05621522  0.04656012  0.08461062  0.94619578 -0.0123551
#  -0.01273037  0.08832292], b: 0.9970172780469101, MSE: 137.50502519032116, VE: 0.5068

# load excel file
df = pd.read_excel('Concrete_Data.xls', sheet_name='Sheet1')

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

X = df[predictors].to_numpy()
y = df[target_value].to_numpy().flatten()

# standardization of features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# normalization of features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# log transform
X_log = np.log(X+1)

# print p-values for data set
X_all = sm.add_constant(X_log)
model = sm.OLS(y, X_all).fit()
print(model.summary())
print(f"\np-values: {model.pvalues}")

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=900, test_size=130, random_state=0)

# add constant for bias
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

model_train = sm.OLS(y_train, X_train).fit()
print(model_train.summary())

y_pred_train = model_train.predict(X_train)
mse_train = mean_squared_error(y_train, y_pred_train)
ve_train = model_train.rsquared
print(f'Train MSE: {mse_train:.4f} and VE: {ve_train:.4f}')

print('\n\n\n Testing Data: \n\n\n')

model_test = sm.OLS(y_test, X_test).fit()
print(model_test.summary())

y_pred_test = model_test.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
ve_test = model_test.rsquared
print(f'Test MSE: {mse_test:.4f} and VE: {ve_test:.4f}')

# plot training data
plt.scatter(y_train, y_pred_train, alpha=0.7, color='blue')
plt.title('Multivariate Raw Training Data: Actual vs Predicted')
plt.xlabel('Actual Compressive Strength')
plt.ylabel('Predicted Compressive Strength')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], lw=2, color='red', linestyle='--')
plt.show()

# plot test data
plt.scatter(y_test, y_pred_test, alpha=0.7, color='green')
plt.title('Multivariate Raw Testing Data: Actual vs Predicted')
plt.xlabel('Actual Compressive Strength')
plt.ylabel('Predicted Compressive Strength')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], lw=2, color='red', linestyle='--')
plt.show()
