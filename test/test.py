import pandas as pd
df=pd.read_excel(r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\dataset cleaned\all_well.xlsx')
df['kick_recognition'] = 111
def feature_rep_mean(window,thrsh_tva,thrsh_gasa,thrsh_spp,thrsh_mfop,p):
    # threshold for each feature std+mean of the feature when the kick is happen
    gasa=window['variation_GASA (mol/mol)'].mean() 
    tva=window['variation_TVA (m3)'].mean() 
    spp=window['variation_SPPA (kPa)'].mean()
    mfop=window['variation_MFOP ((m3/s)/(m3/s))'].mean()
        
    if gasa>=thrsh_gasa and tva>=thrsh_tva and spp<=thrsh_spp and mfop<=thrsh_mfop:
        if p:
            print('all')
            print(tva,gasa,spp,mfop)
        return 1 
    if gasa>=thrsh_gasa and tva>=thrsh_tva and spp<=thrsh_spp:
        if p:
            print('gasa+tva+spp')
            print(tva,gasa,spp,mfop)
        return 1
    if gasa>=thrsh_gasa and tva>=thrsh_tva and mfop<=thrsh_mfop:
        if p:
            print('gasa+tva+mfop')
            print(tva,gasa,spp,mfop)
        return 1
    if gasa>=thrsh_gasa and spp<=thrsh_spp and mfop<=thrsh_mfop:
        if p:
            print('gasa+spp+mfop')
            print(tva,gasa,spp,mfop)
        return 1
    if  tva>=thrsh_tva and spp<=thrsh_spp and mfop<=thrsh_mfop:
        if p:
            print('tva+spp+mfop')
            print(tva,gasa,spp,mfop)
        return 1
    if gasa>=thrsh_gasa and tva>=thrsh_tva :
        if p:
            print('gasa+tva')
            print(tva,gasa,spp,mfop)
        return 1
    if gasa>=thrsh_gasa and spp<=thrsh_spp :
        if p:
            print('gasa+spp')
            print(tva,gasa,spp,mfop)
        return 1
    if gasa>=thrsh_gasa and mfop<=thrsh_mfop :
        if p:
            print('gas+amfop')
            print(tva,gasa,spp,mfop)
        return 1
    if  tva>=thrsh_tva and mfop<=thrsh_mfop :
        if p:
            print('tva+mfop')
            print(tva,gasa,spp,mfop)
        return 1
    if  tva>=thrsh_tva and spp<=thrsh_spp :
        if p:
            print('tva+spp')
            print(tva,gasa,spp,mfop)
        return 1
    if  spp<=thrsh_spp and mfop<=thrsh_mfop:
        if p:
            print('spp+mfop')
            print(tva,gasa,spp,mfop)
        return 1
    if gasa>=thrsh_gasa or tva>=thrsh_tva :
        if p:
            print('gasa or tva')
            print(tva,gasa,spp,mfop)
        return 1
    else :
        return 0
data=df[df['STATUS']==1]
window_size = 180
window_size = window_size//5
ACC = 0
statu_acc = 0
window_size =180
window_size=window_size//5
spp=[]
tva=[]
gasa=[]
mfop=[]
for i in range(len(data['variation_TVA (m3)'])-window_size+1):
        window = data.iloc[i:i+window_size]
        gasa.append(window['variation_GASA (mol/mol)'].mean())
        mfop.append(window['variation_MFOP ((m3/s)/(m3/s))'].mean())
        spp.append(window['variation_SPPA (kPa)'].mean())
        tva.append(window['variation_TVA (m3)'].mean())


for g in gasa:
        for t in tva:
                for s in spp:
                        for m in mfop[59:]:
                                
                                for i in range(len(df)-window_size+1):
                                        p=False
                                        # if i in status_index:
                                        #         p=True
                                        #         print('----------------- {}---------{}'.format(i,window['id'][i]))
                                        window = df.iloc[i:i+window_size]
                                        # window_varation(window,wind_var)       #### i de status a au min avoir un 1 dans un window
                                        # print(feature_rep_mean(window,thrsh_gasa,thrdh_mfop,thrsh_spp,thrsh_tva))
                                        df['kick_recognition'][i:i+window_size] = feature_rep_mean(
                                                window, t, g, s, m,p)
                                ACC=sum((df['STATUS']==df['kick_recognition']))/df.shape[0] #calculate accuracy of algorithm 
                                data=df[df['STATUS']==1]#for get just where status=1
                                statu_acc=sum((data['STATUS']==data['kick_recognition']))/data.shape[0]
               
                                print('accuracy: ',ACC,'accuracy status',statu_acc,t,g,s,m)