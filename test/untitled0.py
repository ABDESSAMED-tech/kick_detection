# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:17:42 2023

@author: ABDESSAMED
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_excel(
    r"C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\dataset\Well-6.xlsx")

df.dtypes
df['kick_recognition'] = 0
df.shape

data = df[df['STATUS'] == 1]
# data.shape
# data.columns

# df[['variation_TVA (m3)','STATUS','TVA (m3)']].plot()
# df[['variation_MFOP ((m3/s)/(m3/s))','STATUS','MFOP ((m3/s)/(m3/s))']].plot()
# df[['variation_MFOA (m3/s)', 'STATUS','MFOA (m3/s)']].plot()
# df[['variation_GASA (mol/mol)', 'STATUS','GASA (mol/mol)']].plot()
# df[['variation_SPPA (kPa)', 'STATUS']].plot()


# diff=data['SPPA (kPa)'].diff()
# data.columns

# sign = np.sign(diff)
# print(sign)

# if all(sign[3487:3487+10] == 1):
#     print('Values are increasing in range of index 2 to 4')
# elif all(sign[3487:3487+10] == -1):
#     print('Values are decreasing in range of index 2 to 4')
# else:
#     print('Values are not all increasing or decreasing in range of index 2 to 4')


# # Show legend
# plt.legend()

# # Show the plot
# plt.show()



cols = ['variation_TVA (m3)', 'variation_SPPA (kPa)',
        'variation_MFOP ((m3/s)/(m3/s))', 'variation_MFOA (m3/s)',
        'variation_GASA (mol/mol)']

min_max_variation_list = {}

for i in cols:
    min_max_variation_list[i] = data[i].sum()


# def feature_rep(window, thresholds):
#     gasa, mfoa, spp, mfop, tva = False, False, False, False, False
#     if window['variation_GASA (mol/mol)'].sum() >= thresholds['variation_GASA (mol/mol)'] :
#         gasa = True
        
#     if window['variation_TVA (m3)'].sum() > thresholds['variation_TVA (m3)']:
#         tva = True
        
#     if window['variation_SPPA (kPa)'].sum() <= thresholds['variation_SPPA (kPa)']:
#         spp = True
        
#     if window['variation_MFOA (m3/s)'].sum() >= thresholds['variation_MFOA (m3/s)']:
#         mfoa = True
       
#     if  window['variation_MFOP ((m3/s)/(m3/s))'].sum() >=thresholds['variation_MFOP ((m3/s)/(m3/s))']:
#         mfop = True

#     return [gasa, mfoa, mfop, spp, tva]

def feature_rep(window):
    gasa, mfoa, spp, mfop, tva = False, False, False, False, False
    if window['variation_GASA (mol/mol)'].sum() >= 0.01:
        gasa = True
        
    if window['variation_TVA (m3)'].sum() > 0:
        tva = True
        
    if window['variation_SPPA (kPa)'].sum() <0:
        spp = True
        
    if window['variation_MFOA (m3/s)'].sum() <=0:
        mfoa = True
       
    if  window['variation_MFOP ((m3/s)/(m3/s))'].sum() <0:
        mfop = True

    return [gasa, mfoa, mfop, spp, tva]


def kick_detection(featurs):
    if featurs[0] and featurs[1]and  featurs[2]and  featurs[3]and featurs[4]:
        print("hhhhhhhhh")
        return 1
        
    else:
        return 0


window_size = 180
window_size = int(window_size // 5)  # convert time to rows
windows = df.groupby((df.index // window_size) + 1)

result = []
for i, batch in windows:
   
    featurs = feature_rep(batch)
    cla = kick_detection(featurs)
    df['kick_recognition'][i:i+window_size+1]=cla
    
    result.append(cla)

    #
set(result)
print(len(result))
ACC=sum((df['STATUS']==df['kick_recognition']))/df.shape[0]
print(ACC)


false_pred_idx = df.index[ df['kick_recognition']!=df['STATUS']]

print(len(false_pred_idx))



# import seaborn as sns 

# dataf=pd.read_excel(r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\dataset\Well-6_01-07-2022.xlsx')
# dataf.dtypes
# dataf.shape
# data=dataf[df['STATUS'] == 1]
# dataf.columns
# cols=['CHKP (kPa)', 'SPM1 (1/s)', 
#        'SQID', 'TVA (m3)',   'MFOP ((m3/s)/(m3/s))', 
#        'MFOA (m3/s)',  'DBTV (m)', 'MFIA (m3/s)', 
#        'MTOA (degC)', 'BPOS (m)', 'MTIA (degC)',
#        'ROPA (m/h)', 'HKLA (N)', 'HKLX (N)',
#        'STKC',  
#        'GASA (mol/mol)',  'SPPA (kPa)', 'RIG_STATE'
#        ]
# len(cols)
# corr = data[cols].corr()


# plt.figure(figsize=(20, 20))  # set figure size
# sns.heatmap(corr, cmap='coolwarm', annot=True)  # create heatmap with annotations
# plt.title('Correlation Matrix')  # add title
# plt.show() 







