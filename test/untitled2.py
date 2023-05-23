# -*- coding: utf-8 -*-
"""
Created on Sat May 20 12:56:52 2023

@author: hp
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_excel(r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\dataset\Well-6.xlsx')
df.shape

index=df[df['MFOP ((m3/s)/(m3/s))']<=0]

print(index.index)