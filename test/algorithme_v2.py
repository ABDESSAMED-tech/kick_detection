# -*- coding: utf-8 -*-
"""
Created on Sun May 14 10:42:35 2023

@author: hp
"""


from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def feature_rep_sum(window):
    gasa, mfoa, spp, mfop, tva = False, False, False, False, False
    if window['variation_GASA (mol/mol)'].sum() <0.42:
        gasa = True
        
    elif window['variation_TVA (m3)'].sum() >=0.2 :
        tva = True
        
    elif window['variation_SPPA (kPa)'].sum() <=0:
        spp = True
        
    # elif window['variation_MFOA (m3/s)'].sum() <=0:
    #     mfoa = True
       
    elif  window['variation_MFOP ((m3/s)/(m3/s))'].sum() <=0:
        mfop = True

    return [gasa, mfop, spp, tva]

def feature_rep_mean(window):
    gasa, mfoa, spp, mfop, tva = False, False, False, False, False
    # threshold for each feature std+mean of the feature when the kick is happen
    if window['variation_GASA (mol/mol)'].mean() >=0.000270+ 0.002598:
        gasa = True
        
    elif window['variation_TVA (m3)'].mean() >=0.011009+0.084261 :
        tva = True
        
    elif window['variation_SPPA (kPa)'].mean() <=-98.107968+368.415950:
        spp = True
        
    # elif window['variation_MFOA (m3/s)'].sum() <=0:
    #     mfoa = True
       
    elif  window['variation_MFOP ((m3/s)/(m3/s))'].mean() <=-0.001468+0.015020:
        mfop = True

    return [gasa, mfop, spp, tva]

def kick_detection(featurs):

    if featurs[0] or (featurs[1]and  featurs[2]or  featurs[3]):
        return 1
    else:
        
        return 0
df=pd.read_excel(r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\dataset\Well-26.xlsx')
df['kick_recognition'] = 111 #just initialzation 
# df.shape
# df.columns
# cols=['TVA (m3)', 'SPPA (kPa)',
#         'MFOP ((m3/s)/(m3/s))', 'GASA (mol/mol)',
#         ]
# cols_variation=['variation_TVA (m3)', 'variation_SPPA (kPa)',
# 'variation_MFOP ((m3/s)/(m3/s))', 'variation_MFOA (m3/s)',
# 'variation_GASA (mol/mol)']
# wind_var={
#     'variation_TVA (m3)':[], 'variation_SPPA (kPa)':[],
#     'variation_MFOP ((m3/s)/(m3/s))':[], 'variation_MFOA (m3/s)':[],
#     'variation_GASA (mol/mol)':[]
    
#     }

#this function for get thresholds 
# def window_varation(window,dic):
    
#     for j in dic.keys() :
#         dic[j].append(window[j].sum())

window_size =120
window_size=window_size//5
for i in range(len(df)-window_size+1):
        window = df.iloc[i:i+window_size]
        # window_varation(window,wind_var)
        df['kick_recognition'][i:i+window_size]=kick_detection(feature_rep_mean(window))
        
        
ACC=sum((df['STATUS']==df['kick_recognition']))/df.shape[0] #calculate accuracy of algorithme
print("Accuracy",ACC)

data=df[df['STATUS']==1]#for get just where status=1
statu_acc=sum((data['STATUS']==data['kick_recognition']))/data.shape[0]
print('status accuracy',statu_acc)
data[['STATUS','kick_recognition']].plot() #plot just where status =1
# df['kick_recognition'].plot()


# df_variation=pd.DataFrame(wind_var)
# data.describe()
# data.shape

# for i in cols_variation:
#     sns.histplot(df_variation[i][14571:14679], kde=True)
#     plt.xlabel('Sum of Variations'+i)
#     plt.show(False)

# for i in cols_variation:
#     print(i)
#     print(data[i].describe())
    


        
