import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# df=pd.read_excel(r"C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\data\Well-6_01-07-2022.xlsx")
# def sig_var(df,target):
# #    df = df.reset_index()
#    df['variation_'+target] = 0
#    for index, row in df.iterrows():
#        if index == 0:
#            prev_row = row
#            df['variation_'+target][index] = df[target][index]
#            continue
#        df['variation_'+target][index] = row[target] - prev_row[target]
#        prev_row = row
#    return df


data=pd.read_excel(r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\data\Well-6_01-07-2022.xlsx')

def sig_var(df,target):
    df['Varaiton_'+target]=df[target].diff()
    return df

cols=['TVA (m3)', 'SPPA (kPa)', 'MFOA (m3/s)',
       'MFOP ((m3/s)/(m3/s))', 'GASA (mol/mol)']

for i in cols:
    sig_var(data,i)



# data=sig_var(df,'TVA (m3)')

# data=sig_var(data,'SPPA (kPa)')
# data=sig_var(data,'MFOP ((m3/s)/(m3/s))')
# data=sig_var(data,'MFOA (m3/s)')

# data=sig_var(data,'GASA (mol/mol)')

# data.to_excel(r"C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\dataset\Well-6.xlsx")
df=pd.read_excel(r"C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\dataset\Well-6.xlsx")
df.columns
df.dtypes

df.shape
# Index(['Unnamed: 0', 'level_0', 'index', 'TVA (m3)', 'SPPA (kPa)',
#        'MFOA (m3/s)', 'MFOP ((m3/s)/(m3/s))', 'GASA (mol/mol)', 'STATUS',
#        'Unnamed: 41', 'variation_TVA (m3)', 'variation_SPPA (kPa)',    
#        'variation_MFOP ((m3/s)/(m3/s))', 'variation_MFOA (m3/s)',      
#        'variation_GASA (mol/mol)'],
#       dtype='object')




def feature_rep(window):
    gasa,mfoa,spp,mfop,tva=False,False,False,False,False
    
    tva_thresh = window['variation_TVA (m3)'].mean()
    mfop_thresh =  window['variation_MFOP ((m3/s)/(m3/s))'].mean()
    mfoa_thresh =  window['variation_MFOA (m3/s)'].mean()
    spp_thresh =  window['variation_SPPA (kPa)'].mean()
    gas_thresh =  window['variation_GASA (mol/mol)'].mean()
    
    if window['variation_GASA (mol/mol)'].sum()>gas_thresh:
        gasa=True
    if window['variation_TVA (m3)'].sum()>0:
        tva=True
    if window['variation_SPPA (kPa)'].sum()>0:
        spp=True
    if window['variation_MFOA (m3/s)'].sum()>=0:
        mfoa=True
    if window['variation_MFOP ((m3/s)/(m3/s))'].sum()<0:
        mfop= True
        
    
    return [gasa,mfoa,mfop,spp,tva]

def kick_detection(featurs):
    return all(featurs)
    

window_size = 120
window_size = int(window_size // 5)# convert time to rows
windows = df.groupby((df.index // window_size) + 1)


result=[]
for i, batch in windows:
    # print(f'--------window number {i}--------')
    # print(batch.head(5))
    featurs=feature_rep(batch)
    cla=kick_detection(featurs)
    print(cla)
    result.append(cla) 
    
    #
print(len(result)) 
