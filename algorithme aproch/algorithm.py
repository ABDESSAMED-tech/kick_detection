import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def feature_rep_mean(window, thrsh_tva, thrsh_gasa, thrsh_spp, thrsh_mfop, p):
    # threshold for each feature mean of the feature when the kick is happen
    gasa = window['variation_GASA (mol/mol)'].mean()
    tva = window['variation_TVA (m3)'].mean()
    spp = window['variation_SPPA (kPa)'].mean()
    mfop = window['variation_MFOP ((m3/s)/(m3/s))'].mean()

    # if gasa>=thrsh_gasa and tva>=thrsh_tva and spp<=thrsh_spp:

    #     return 1

    if (gasa > thrsh_gasa or tva > thrsh_tva and mfop < thrsh_mfop) and (gasa >thrsh_gasa and spp < thrsh_spp and mfop < thrsh_mfop):
        # print(tva,gasa,spp,mfop)
        return 1
    
    # if (gasa > thrsh_gasa or tva > thrsh_tva and mfop < thrsh_mfop) and (gasa >thrsh_gasa and spp < thrsh_spp and mfop < thrsh_mfop):
    #     return 1

    # if (gasa > thrsh_gasa or tva > thrsh_tva and mfop < thrsh_mfop) and (gasa >thrsh_gasa or spp < thrsh_spp and mfop < thrsh_mfop):
    #     return 0
    # if gasa>=thrsh_gasa and tva > thrsh_tva and mfop<=thrsh_mfop:
    #     if p:
    #         print('gasa+spp+mfop')
    #         print(tva,gasa,spp,mfop)
    #     return 1
    # if  tva>=thrsh_tva and gasa > thrsh_gasa and mfop<=thrsh_mfop:
    #     return 1

    else:
        return 0
    
file_path = r"C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\dataset cleaned\all_data_without_26.xlsx"
df = pd.read_excel(file_path)
df['kick_recognition'] = 111# just init for column for use it later




""" this the best thresholds with accuracy """
# thrsh_tva, thrsh_gasa, thrsh_spp, thrsh_mfop = 0.005, -0.0049, 1, -0.000345541 #accuracy:  0.9165375463449726 accuracy status 0.3329268292682927
# thrsh_tva, thrsh_gasa, thrsh_spp, thrsh_mfop = 0.005, -0.0049, 2, -0.000345541 #accuracy:  0.9088594986442367 accuracy status 0.36097560975609755
thrsh_tva, thrsh_gasa, thrsh_spp, thrsh_mfop =0.005, -0.0049, 0.1, -0.000345541#accuracy:  0.9259172154280338 accuracy status 0.3121951219512195
# thrsh_tva, thrsh_gasa, thrsh_spp, thrsh_mfop = 0.00843, 0.4, 11,-0.00023


window_size = 180
window_size = window_size//5


positive=0
train_data=[]
for i in range(len(df)-window_size+1):
    p=False
    window = df.iloc[i:i+window_size]
    # window_varation(window,wind_var)       #### i de status a au min avoir un 1 dans un window
    # print(feature_rep_mean(window,thrsh_gasa,thrdh_mfop,thrsh_spp,thrsh_tva))
    df['kick_recognition'][i:i+window_size+1] = feature_rep_mean(
        window, thrsh_tva, thrsh_gasa, thrsh_spp, thrsh_mfop, p)
    if window['kick_recognition'].sum()>0 and window['STATUS'].sum()>0:
        positive+=1

print(positive/(len(df)-window_size+1))
ACC = sum((df['STATUS'] == df['kick_recognition'])) / \
    df.shape[0]  # calculate accuracy of algorithm
data = df[df['STATUS'] == 1]  # for get just where status=1
statu_acc = sum((data['STATUS'] == data['kick_recognition']))/data.shape[0]
print('accuracy: ', ACC, 'accuracy status', statu_acc)