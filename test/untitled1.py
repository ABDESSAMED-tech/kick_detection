# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:41:16 2023

@author: hp
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_excel(r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\dataset\Well-6.xlsx')

df.columns
duplicates=df[df.duplicated()]

cols=['TVA (m3)', 'SPPA (kPa)',
       'MFOA (m3/s)', 'MFOP ((m3/s)/(m3/s))', 'GASA (mol/mol)']
print(len(duplicates))


def get_outlier(feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df[feature][(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    return outliers

df.shape

outlier_gasa=get_outlier('MFOP ((m3/s)/(m3/s))')
print(len(outlier_gasa))
# Define the lower and upper bounds for outliers
df = df[~df.index.isin(outlier_gasa.index)]

df.reset_index()