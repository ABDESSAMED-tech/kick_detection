import streamlit as st
import pandas as pd
from tabulate import tabulate
st.set_page_config(page_title="Data summary", page_icon='📊')

st.title("Data summary ")
# df=st.session_state['df']


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
   
    
if 'df' in st.session_state:
    df = st.session_state['df']
    stats()
else:
    st.warning('Please load the dataset')
    
    
footer = '''
<style>
.footer {
    position: fixed;
    left: 20;
    bottom: 0;
    width: 100%;
    background-color: #f8f9fa;
    color: #333333;
    text-align: left;
    padding: 10px 20px;
    box-sizing: border-box;
}

@media screen and (max-width: 600px) {
    .footer {
        text-align: center;
        position: static;
    }
}
</style>

<div class="footer">
    <p>This app is developed by BOULARIACHE Abdessamed and TAZIR Mohamed Reda.</p>
</div>
'''

# Render the footer using the st.beta_container() function
st.container().write(footer, unsafe_allow_html=True)

# Render the footer using the st.beta_container() function
st.container().write(footer, unsafe_allow_html=True)