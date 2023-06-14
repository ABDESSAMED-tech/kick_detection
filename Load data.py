import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly_express as px 
st.set_page_config(page_title="Kick detection using ML", page_icon=":rocket:",)
@st.cache_data()
def load_data(file):
    data=pd.read_excel(file)
    return data
st.sidebar.caption('leaod the data before using the models')

st.title('THE APPLICATION OF MACHINE LEARNING IN KICK DETECTION AND PREDICTION DURING DRILLING OPERATIONS')
st.sidebar.title('Load data')
uploaded_file = st.sidebar.file_uploader('Upload your file here ', 'xlsx')

if uploaded_file:
    df = load_data(uploaded_file)
    st.session_state['df']=df
else:
    st.warning('Please upload a dataset !!')
    
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
    <p>This app is developed by BOULARIACHE Abdessamed and TAZIR Mouhamed Reda.</p>
</div>
'''

st.container().write(footer, unsafe_allow_html=True)

