import streamlit as st
import pandas as pd
st.set_page_config(page_title="Kick detection using ML", page_icon='📊')

st.title("Data summary ")
df=st.session_state['df']

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
    

stats()
    