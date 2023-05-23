from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def feature_rep_mean(window,thrsh_tva,thrsh_gasa):
    gasa, mfoa, spp, mfop, tva = 0, 0, 0, 0, 0
    # threshold for each feature std+mean of the feature when the kick is happen
    if window['GASA (mol/mol)'].mean() >=thrsh_gasa:
        gasa = 1
        
    elif window['TVA (m3)'].mean() >=thrsh_tva :
        tva = 1
        
    
    return [gasa, tva]
def kick_detection(featurs):
    
    if featurs[0] or  featurs[1]:
        return 1
    else:
        
        return 0
df=pd.read_excel(r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\dataset\Well-15.xlsx')
df['kick_recognition'] = 111 #just initi
thrsh_tva,thrsh_gasa=  56.6,0.7
window_size =180
window_size=window_size//5
ACC=0
statu_acc=0
while thrsh_tva<58:
        
                for i in range(len(df)-window_size+1):
                        window = df.iloc[i:i+window_size]
                # window_varation(window,wind_var)       #### i de status a au min avoir un 1 dans un window
                        #print(feature_rep_mean(window,thrsh_gasa,thrdh_mfop,thrsh_spp,thrsh_tva))
                        df['kick_recognition'][i:i+window_size]=kick_detection(feature_rep_mean(window,thrsh_tva,thrsh_gasa))
                ACC=sum((df['STATUS']==df['kick_recognition']))/df.shape[0] #calculate accuracy of algorithme
                data=df[df['STATUS']==1]#for get just where status=1
                statu_acc=sum((data['STATUS']==data['kick_recognition']))/data.shape[0]
                
                print(ACC,statu_acc,thrsh_gasa,thrsh_tva)
        
                thrsh_tva=thrsh_tva+0.01