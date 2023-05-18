# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:52:43 2023

@author: hp
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def feature_rep_mean(window,thrsh_gasa,thrdh_mfop,thrsh_spp,thrsh_tva):
    gasa, mfoa, spp, mfop, tva = 0, 0, 0, 0, 0
    # threshold for each feature std+mean of the feature when the kick is happen
    if window['GASA (mol/mol)'].mean() >=thrsh_gasa:
        gasa = 1
        
    elif window['TVA (m3)'].mean() >=thrsh_tva :
        tva = 1
        
    elif window['SPPA (kPa)'].mean() <=thrsh_spp:
        spp = 1
        
    # elif window['variation_MFOA (m3/s)'].sum() <=0:
    #     mfoa = True
       
    elif  window['MFOP ((m3/s)/(m3/s))'].mean() <=thrdh_mfop:
        mfop = 1
    #print (gasa, mfop, spp, tva)
    return [gasa, mfop, spp, tva]

def kick_detection(featurs):
    
    if featurs[0] and  (featurs[1]and  featurs[2]and  featurs[3]):
        print('cond1',featurs)
        return 1
    elif  featurs[0] and featurs[1]and  featurs[2]:
        print('cond2',featurs)
        return 1
    elif  featurs[0] and featurs[1]and  featurs[3]:
        print('cond3',featurs)
        return 1
    elif  featurs[0] and featurs[2]and  featurs[3]:
        print('cond4',featurs)
        return 1
    elif  featurs[1] and featurs[2]and  featurs[3]:
        print('cond5',featurs)
        return 1
    elif  (featurs[0] and featurs[1]) or(featurs[0] and featurs[2])or (featurs[0] and featurs[3])  :
        print('cond6',featurs)
        return 1
    elif  (featurs[2] and featurs[3]) or(featurs[1] and featurs[2])or (featurs[1] and featurs[3])  :
        print('cond7',featurs)
        return 1
    elif  featurs[0] or ( featurs[1]and  featurs[2]or  featurs[3])  :
        # print('cond8',featurs)
        return 1
    else:
        
        return 0
def window_varation(window,dic): 
    for j in dic.keys() :
        dic[j].append(window[j].min())
    
wind_var={
    'variation_TVA (m3)':[], 'variation_SPPA (kPa)':[],
    'variation_MFOP ((m3/s)/(m3/s))':[], 'variation_MFOA (m3/s)':[],
    'variation_GASA (mol/mol)':[]
    
    }
    
df=pd.read_excel(r'C:\Users\DropZone\Desktop\Nouveau\kick_detection\dataset\Well-6.xlsx')
df['kick_recognition'] = 111 #just initialzation

cols_variation=['variation_TVA (m3)', 'variation_SPPA (kPa)',
'variation_MFOP ((m3/s)/(m3/s))', 'variation_MFOA (m3/s)',
'variation_GASA (mol/mol)']

window_size =180
window_size=window_size//5

tva=[-0.200000,-0.1]
spp=[-1896.058256,-6.894757]
mfop=[-0.060000,0]
gasa=[ -0.006400,-0.000100]

#thrsh_gasa,thrdh_mfop,thrsh_spp,thrsh_tva= 0.056960,0.110917,4987.867887,73.482569
thrsh_gasa,thrdh_mfop,thrsh_spp,thrsh_tva= 0,0,0,0

train_data=[]
for i in range(len(df)-window_size+1):
        window = df.iloc[i:i+window_size]
       # window_varation(window,wind_var)       #### i de status a au min avoir un 1 dans un window
        train=feature_rep_mean(window,thrsh_gasa,thrdh_mfop,thrsh_spp,thrsh_tva)
        train.append(df['STATUS'][i])
        train_data.append(train)
        #print(feature_rep_mean(window,thrsh_gasa,thrdh_mfop,thrsh_spp,thrsh_tva))
        df['kick_recognition'][i:i+window_size]=kick_detection(feature_rep_mean(window,thrsh_gasa,thrdh_mfop,thrsh_spp,thrsh_tva))
print(train_data)     
        
ACC=sum((df['STATUS']==df['kick_recognition']))/df.shape[0] #calculate accuracy of algorithme
#print("Accuracy",ACC)

data=df[df['STATUS']==1]#for get just where status=1
statu_acc=sum((data['STATUS']==data['kick_recognition']))/data.shape[0]
#print('status accuracy',statu_acc)
df[['STATUS','kick_recognition']].plot()



#for i in data:
   # print(i,data[i].describe())

# data.plot()
# # data.describe()
# data.shape