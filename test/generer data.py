# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:23:38 2023

@author: hp
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def feature_rep_mean(window,thrsh_gasa,thrdh_mfop,thrsh_spp,thrsh_tva):
    gasa, mfoa, spp, mfop, tva = False, False, False, False, False
    # threshold for each feature std+mean of the feature when the kick is happen
    if window['variation_GASA (mol/mol)'].mean() >=thrsh_gasa:
        gasa = True
        
    elif window['variation_TVA (m3)'].mean() >=thrsh_tva :
        tva = True
        
    elif window['variation_SPPA (kPa)'].mean() <=thrsh_spp:
        spp = True
        
    # elif window['variation_MFOA (m3/s)'].sum() <=0:
    #     mfoa = True
       
    elif  window['variation_MFOP ((m3/s)/(m3/s))'].mean() <=thrdh_mfop:
        mfop = True

    return [gasa, mfop, spp, tva]
def kick_detection(featurs):
    
    if featurs[0] and  (featurs[1]and  featurs[2]and  featurs[3]):
        
        return 1
    elif  featurs[0] and featurs[1]and  featurs[2]:
       
        return 1
    elif  featurs[0] and featurs[1]and  featurs[3]:
        return 1
    elif  featurs[0] and featurs[2]and  featurs[3]:
        
        return 1
    elif  featurs[1] and featurs[2]and  featurs[3]:
        
        return 1
    elif  (featurs[0] and featurs[1]) or(featurs[0] and featurs[2])or (featurs[0] and featurs[3])  :
        
        return 1
    elif  (featurs[2] and featurs[3]) or(featurs[1] and featurs[2])or (featurs[1] and featurs[3])  :
        return 1
    elif  featurs[0] or ( featurs[1]and  featurs[2]or  featurs[3])  :
        # print('cond8',featurs)
        return 1
    else:
        
        return 0
df=pd.read_excel(r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\dataset\all.xlsx')
df['kick_recognition'] = 111 #just initialzation


thrsh_tva,thrsh_spp,thrdh_mfop,thrsh_gasa= 0.144,0.533865,0.000003,0.1214953

train_data=[]
window_size =180
window_size=window_size//5
for i in range(len(df)-window_size+1):
        window = df.iloc[i:i+window_size]
       # window_varation(window,wind_var)       #### i de status a au min avoir un 1 dans un window
        # train=feature_rep_mean(window,thrsh_gasa,thrdh_mfop,thrsh_spp,thrsh_tva)
        # train.append(df['STATUS'][i])
        # train_data.append(train)
        #print(feature_rep_mean(window,thrsh_gasa,thrdh_mfop,thrsh_spp,thrsh_tva))
        df['kick_recognition'][i:i+window_size]=kick_detection(feature_rep_mean(window,thrsh_gasa,thrdh_mfop,thrsh_spp,thrsh_tva))
# print(train_data)     
        
ACC=sum((df['STATUS']==df['kick_recognition']))/df.shape[0] #calculate accuracy of algorithme
print("Accuracy",ACC)

data=df[df['STATUS']==1]#for get just where status=1
statu_acc=sum((data['STATUS']==data['kick_recognition']))/data.shape[0]
print('status accuracy',statu_acc)
df[['STATUS','kick_recognition']].plot()
cols_variation=['variation_TVA (m3)', 'variation_SPPA (kPa)',
'variation_MFOP ((m3/s)/(m3/s))', 'variation_MFOA (m3/s)',
'variation_GASA (mol/mol)']
for i in data[cols_variation]:
    print(i,df[i].describe())

data['variation_TVA (m3)'].mean()/5
