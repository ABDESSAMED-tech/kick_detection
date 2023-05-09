import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly_express as px 
st.set_page_config(page_title="Kick detection using ML", page_icon=":rocket:",)
@st.cache_data()
def load_data(file):
    data=pd.read_excel(file)
    return data
def Home():
    st.text("""A kick occurs when uncontrolled flow of formation or other fluids takes place. The fluid could flow to
        the surface (surface blowout) or to an exposed formation with lower pressure (underground blowout). It is
        very crucial to detect and control a kick as soon as possible. Detecting kicks in early stages gives the crew
        additional time to control it preventing loss of well control (LWC) and therefore having a safer and more
        efficient drilling operation.
        In order to maximize operational safety, a reliable automated method to detect kicks before it becomes
        critical is an industry priority. Some companies have started installing more sensors at the rigs to help detect
        kicks, others have tried to automate the prediction with the current system they have, and few have combined
        both methods. As for the prediction of kicks, there are two ways of doing it: physics approach and data
        mining approach. Both approaches look at historical data of loss control incident then using their finding
        try to predict when a future kick will happen. The beauty with the physics approach is it makes sense and
        easy to understand, explain, and usually involves a flow chart of few if-statements. Whereas data mining
        approach takes in the whole data set, crunches it together, and then come up with a model using artificial
        intelligence with the result expressed as a probability. """)



st.title('The application of machine learning for kick detection ')
st.sidebar.title('Navigation')
uploaded_file = st.sidebar.file_uploader('Upload your file here ', 'xlsx')
options = st.sidebar.radio('Pages', options=
                           ["Home"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.session_state['df']=df
else:
    st.warning('Please upload a dataset !!')