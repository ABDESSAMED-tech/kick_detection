import streamlit as st
import pandas as pd
from tabulate import tabulate
st.set_page_config(page_title="Kick detection using ML", page_icon='📊')

st.title("Data summary ")
df=st.session_state['df']
cols=['index', 'CHKP (kPa)', 'SPM1 (1/s)', 'SKNO', 'SPM2 (1/s)', 'SPM3 (1/s)',
       'SQID', 'TVA (m3)', 'DATE', 'TIME', 'MFOP ((m3/s)/(m3/s))', 'ACTC',
       'MFOA (m3/s)', 'DBTM (m)', 'DBTV (m)', 'MFIA (m3/s)', 'DMEA (m)',
       'MDIA (kg/m3)', 'DVER (m)', 'MTOA (degC)', 'BPOS (m)', 'MTIA (degC)',
       'ROPA (m/h)', 'MCOA (S/m)', 'HKLA (N)', 'MCIA (S/m)', 'HKLX (N)',
       'STKC', 'WOBA (N)', 'WOBX (N)', 'DRTM (m)', 'TQA (N.m)', 'TQX (N.m)',
       'GASA (mol/mol)', 'RPMA (rad/s)', 'SPPA (kPa)', 'RIG_STATE',
       'MDOA (kg/m3)', 'ROPI (m/s)', 'STATUS']

def stats():
    st.header('Data Statistics ')
    st.write(df.describe())
    st.header('data Header')
    st.write(df.columns)
    st.header("Missing values")
    data = dict(df.isna().sum())
    st.table(data)
    st.write('Total Missing values is :'+str(df.isna().sum().sum()))
    # print(df.head(10))
   
    
 
    st.header('Description de données')
    st.write(cols)
    st.write(str(df[cols].dtypes))
    

stats()
    