import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
df=pd.read_excel(r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\data\All_data.xlsx')
df.shape
df.columns
df['MFOA (m3/s)'].head()
cols=['TVA (m3)', 'SPPA (kPa)', 'MFOA (m3/s)',
       'MFOP ((m3/s)/(m3/s))', 'GASA (mol/mol)','STATUS']

df_p=df[cols].astype(float)
min_max_scaler = MinMaxScaler()

df_normalized_minmax = pd.DataFrame(min_max_scaler.fit_transform(df_p), columns=df_p.columns)
for i in cols:
    df_normalized_minmax[[i,'STATUS']].plot(figsize=(10, 6))  # create the plot
    plt.legend(loc='upper left')  # add a legend to the plot
    plt.title('line plot  ')  # add a title to the plot
    plt.xlabel('X-axis label')  # add an X-axis label
    plt.ylabel('Y-axis label')  # add a Y-axis label
    plt.show()  # display the plot


#for show all plot 
fig, axs = plt.subplots(nrows=6, ncols=1, figsize=(25, 15))# create a 2x2 grid of subplots
k=0
for i in cols:
    axs[k].plot(df[i])
    # axs[k].set_title('plot of {}'.format(i))
    k=k+1
k=0   
fig.suptitle('Title of the figure')
plt.show()