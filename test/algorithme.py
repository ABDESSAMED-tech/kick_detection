import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_excel(r"C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\data\Well-6_01-07-2022.xlsx")
spp=df['SPPA (kPa)'].to_numpy()
tva=df['TVA (m3)'].to_numpy()
mfoa=df['MFOA (m3/s)'].to_numpy()
mfop=df['MFOP ((m3/s)/(m3/s))'].to_numpy()
gasa=df['GASA (mol/mol)'].to_numpy()
data={'TVA':[],'SPP':[],'MFOA':[],'MFOP':[],"GASA":[],'Kick':[]}
w=60
for i in range(1,len(spp)-(len(spp)%w)-1):
    sub_spp,sub_tva,sub_mfoa,sub_mfop=0,0,0,0
    for j in range(i,i+w-1): 
         sub_spp=sub_spp+spp[j]-spp[j-1]
         sub_tva=sub_tva+tva[j]-tva[j-1]
         sub_mfoa=sub_mfoa+mfoa[j]-mfoa[j-1]
         sub_mfop=sub_mfop+mfop[j]-mfop[j-1]
    #(well 15,0.03) (well 6,0.119)
    if sub_spp<0 and sub_tva>0 and  sub_mfoa>=0 and sub_mfop<=0 and gasa[i]>0.119:
        print('----------{}----kick----------'.format(i))
        data['TVA'].append(tva[i])
        data['MFOA'].append(mfoa[i])
        data['MFOP'].append(mfop[i])
        data['SPP'].append(spp[i])
        data['GASA'].append(gasa[i])
        data['Kick'].append(1)
        
        
        # print('there is a kick ',sub_spp,sub_tva,sub_mfoa,sub_mfop)
    else:
        data['TVA'].append(tva[i])
        data['MFOA'].append(mfoa[i])
        data['MFOP'].append(mfop[i])
        data['SPP'].append(spp[i])
        data['GASA'].append(gasa[i])
        data['Kick'].append(0)
    
data_frame=pd.DataFrame(data)

data_frame.shape
data_frame['Kick'].sum()

   