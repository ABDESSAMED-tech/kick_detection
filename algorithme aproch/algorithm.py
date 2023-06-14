import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def add_variation_column(df, column_name):
    variations = [None]  # Initialize the list of variations with None for the first row
    for i in range(1, len(df)):
        diff = df[column_name].iloc[i] - df[column_name].iloc[i-1]
        variations.append(diff)
    df[f'variation_{column_name}'] = variations
    return df

def feature_rep(window, thrsh_tva, thrsh_gasa, thrsh_spp, thrsh_mfop, p):
    # threshold for each feature mean of the feature when the kick is happen
    gasa = window['GASA (mol/mol)'].mean()
    tva = window['variation_TVA'].mean()
    spp = window['variation_SPPA'].mean()
    mfop = window['variation_MFOP'].mean()
# gasa pas de variasion
    detect_gasa, detect_spp, detect_tva, detect_mfop = False, False, False, False
    l = []
    # if  gasa > thrsh_gasa and tva > thrsh_tva  :  # well 19
    #     l = [int(gasa > thrsh_gasa), int(tva > thrsh_tva),
    #          int(mfop < thrsh_mfop), int(spp < thrsh_spp)]
    #     return 1,l
    # elif tva > thrsh_tva and mfop < thrsh_mfop and gasa > thrsh_gasa:  # well 19
    #     l = [int(gasa > thrsh_gasa), int(tva > thrsh_tva),
    #          int(mfop < thrsh_mfop), int(spp < thrsh_spp)]

    #     return 1, l
    # elif gasa > thrsh_gasa and tva > thrsh_tva and mfop < thrsh_mfop:  # well 17
    #     l = [int(gasa > thrsh_gasa), int(tva > thrsh_tva),
    #          int(mfop < thrsh_mfop), int(spp < thrsh_spp)]

    #     return 1, l
    # elif gasa > thrsh_gasa and tva > thrsh_tva and spp < thrsh_spp:  # well 15
    #     l = [int(gasa > thrsh_gasa), int(tva > thrsh_tva),
    #          int(mfop < thrsh_mfop), int(spp < thrsh_spp)]

    #     return 1, l
    if gasa > thrsh_gasa and mfop < thrsh_mfop and spp < thrsh_spp:  # well 6,8
        l = [int(gasa > thrsh_gasa), int(tva > thrsh_tva),
             int(mfop < thrsh_mfop), int(spp < thrsh_spp)]
        return 1,l

    #     return 1, l
    # elif gasa > thrsh_gasa and tva > thrsh_tva and mfop < thrsh_mfop and spp < thrsh_spp:  # well 26
    #     l = [int(gasa > thrsh_gasa), int(tva > thrsh_tva),
    #          int(mfop < thrsh_mfop), int(spp < thrsh_spp)]
    #     return 1, l
    else:

        l = [int(gasa > thrsh_gasa), int(tva > thrsh_tva),
             int(mfop < thrsh_mfop), int(spp < thrsh_spp)]
        return 0, l
    
    
df=pd.read_excel(r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\dataset\8.xlsx')

df['kick_recognition'] = 111# just init for column for use it later

cols11=['TVA (m3)', 'SPPA (kPa)', 'MFOP ((m3/s)/(m3/s))',
       'GASA (mol/mol)']
data=df[cols11].copy()
for column_name in cols11:
    data=add_variation_column(data, column_name)

data=data.dropna()
data['STATUS']=df['STATUS'][1:]
data = data.rename(columns={'variation_GASA (mol/mol)': 'variation_GASA'})
data = data.rename(columns={'variation_SPPA (kPa)':'variation_SPPA'})
data = data.rename(columns={'variation_MFOP ((m3/s)/(m3/s))': 'variation_MFOP'})
data = data.rename(columns={'variation_TVA (m3)': 'variation_TVA'})

""" this the best thresholds with accuracy """
# # thrsh_tva, thrsh_gasa, thrsh_spp, thrsh_mfop = 0.005, -0.0049, 1, -0.000345541 #accuracy:  0.9165375463449726 accuracy status 0.3329268292682927
# # thrsh_tva, thrsh_gasa, thrsh_spp, thrsh_mfop = 0.005, -0.0049, 2, -0.000345541 #accuracy:  0.9088594986442367 accuracy status 0.36097560975609755
# thrsh_tva, thrsh_gasa, thrsh_spp, thrsh_mfop =0.005, -0.0049, 0.1, -0.000345541#accuracy:  0.9259172154280338 accuracy status 0.3121951219512195
# # thrsh_tva, thrsh_gasa, thrsh_spp, thrsh_mfop = 0.00843, 0.4, 11,-0.00023


window_size = 180
window_size = window_size//5


data['kick_recognition']=111
thrsh_tva, thrsh_gasa, thrsh_spp, thrsh_mfop =1, 0.035, 0.1, 0#well 6,8
# thrsh_tva, thrsh_gasa, thrsh_spp, thrsh_mfop =0, 0.03, 0, 0.1#well 26
# thrsh_tva, thrsh_gasa, thrsh_spp, thrsh_mfop =0.015, 0.08, -1, 0#15
# thrsh_tva, thrsh_gasa, thrsh_spp, thrsh_mfop =0.001, 0.03, -1, 0#17
# thrsh_tva, thrsh_gasa, thrsh_spp, thrsh_mfop =0.01, 0.03, -0.08,0#19
# thrsh_tva, thrsh_gasa, thrsh_spp, thrsh_mfop =0.01, 0.1,0.01,0#29
# thrsh_tva, thrsh_gasa, thrsh_spp, thrsh_mfop = 0.1,0.1,0, 0.005#29
window_size = 180
window_size = window_size//5
positive=0
train_data=[]
d=[]
for i in range(len(data)-window_size+1):
    p=False
    window = data.iloc[i:i+window_size]
    
    data['kick_recognition'][i:i+window_size+1],b = feature_rep(
        window, thrsh_tva, thrsh_gasa, thrsh_spp, thrsh_mfop, p)
    
    b.append(data.iloc[i,-1:][0]) #append df[kick_recognitin][i]
    train_data.append(b)
    
ACC = sum((data['STATUS'] == data['kick_recognition'])) / \
    data.shape[0]  # calculate accuracy of algorithm
data_kik = data[data['STATUS'] == 1]  # for get just where status=1
statu_acc = sum((data_kik['STATUS'] == data_kik['kick_recognition']))/data_kik.shape[0]
print('accuracy: ', ACC, 'accuracy status', statu_acc)


cols=['GASA (mol/mol)','TVA (m3)','MFOP ((m3/s)/(m3/s))', 'SPPA (kPa)' ,
       'detection de kick']
dftest = pd.DataFrame(train_data, columns=cols)
dftest['STATUS']=data['STATUS']
dftest['detection de kick'].value_counts()
dftest.plot(subplots=True)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,roc_auc_score

print(confusion_matrix(dftest['STATUS'], dftest['detection de kick']))
print("Accuracy:", accuracy_score(dftest['STATUS'], dftest['detection de kick']))
print("Precision:", precision_score(dftest['STATUS'], dftest['detection de kick']))
print("Recall:", recall_score(dftest['STATUS'], dftest['detection de kick']))
print("F1 Score:",f1_score(dftest['STATUS'], dftest['detection de kick']))
