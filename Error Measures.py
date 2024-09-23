# Becky Shofner
# CSE514A Data Mining Programming Assignment 1B

# pip install xlrd

import pandas as pd
import numpy as np

# load excel file
df = pd.read_excel('Concrete_Data.xls', sheet_name='Sheet1')
print(df.head())
print(df.columns)

# create copy to normalize variables
df_normalized = df.copy()

for column in df_normalized.columns:
    # normalize all variables but response variable
    if column != 'Concrete compressive strength(MPa, megapascals) ':
        df_normalized[column] = df_normalized[column] / df_normalized[column].abs().max()

print(df_normalized.head())

# create copy to normalize variables
# df_normalized = df.copy()
#
# for column in df_normalized.columns:
#     # normalize all variables but response variable
#     if column != 'Concrete compressive strength(MPa, megapascals) ':
#         df_normalized[column] = df_normalized[column] / df_normalized[column].abs().max()
#
# print(df_normalized.head())


# y = df_normalized['Concrete compressive strength(MPa, megapascals) '].to_numpy()
#
# # List of predictor variables
# predictors = ['Cement (component 1)(kg in a m^3 mixture)',
#               'Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
#               'Fly Ash (component 3)(kg in a m^3 mixture)',
#               'Water  (component 4)(kg in a m^3 mixture)',
#               'Superplasticizer (component 5)(kg in a m^3 mixture)',
#               'Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
#               'Fine Aggregate (component 7)(kg in a m^3 mixture)',
#               'Age (day)']
#
# for predictor in predictors:
#     X = df_normalized[predictor].to_numpy()
