import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,roc_auc_score
import numpy as np
import pandas as pd
import plotly_express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from utils import Knn_algorithme,LSTM_algorithm,SVM_algorithme
df = st.session_state['df']
st.title("Machine Learning Algorithme")


options = st.radio("Select an algorithme  ", options=[
                   'K-Nearest Neighbors (KNN) ',
                   'Support Vector Machine (SVM) ',
                   'LSTM Long-Short-Term-Memory (LSTM)',
                   ])


def attribute_selection(df,option):
    if option=='LSTM Long-Short-Term-Memory (LSTM)':
         with st.form("attribute_form"):
            selected_attributes = st.multiselect("Select attributes", df.columns)
            target_att = st.selectbox('Select target attribute', df.columns)
            epoch = st.slider('Select number of  epochs', min_value=2, max_value=50)
            test_size=st.slider('Select the test size', min_value=1, max_value=100,step=1)
            window_size=st.slider('Select the window size', min_value=12, max_value=100,step=1)

            submit_button = st.form_submit_button(label='Run')
            
            return selected_attributes, target_att, epoch,window_size,test_size
        
    if option=='K-Nearest Neighbors (KNN)':
        with st.form("attribute_form"):
            selected_attributes = st.multiselect("Select attributes", df.columns)
            target_att = st.selectbox('Select target attribute', df.columns)
            k = st.slider('Select value of K-Neighbors', min_value=2, max_value=50)
            test_size=st.slider('Select the test size', min_value=1, max_value=100,step=1)
            window_size=st.slider('Select the window size', min_value=12, max_value=100,step=1)

            submit_button = st.form_submit_button(label='Run')

        # Return the selected attributes
        return selected_attributes, target_att, k,test_size,window_size
    if option=='Support Vector Machine (SVM) ':
        with st.form("attribute_form"):
            selected_attributes = st.multiselect("Select attributes", df.columns)
            target_att = st.selectbox('Select target attribute', df.columns)
            kernel = st.selectbox('Select kernel ', options=['rbf','linear'] )
            window_size=st.slider('Select the window size', min_value=12, max_value=100,step=1)
            c = st.slider('Select a range of Regularization parameter (C)', 0.1, 100.0 )
            gama= st.slider('Select a range of Paramètre gamma (gamma)', 0.01, 100.0)
            test_size=st.slider('Select the test size', min_value=1, max_value=100,step=1)
            

            submit_button = st.form_submit_button(label='Run')

        # Return the selected attributes
        return selected_attributes, target_att, kernel,c,gama,test_size,window_size

def KNN_algorithme(df):
    option='K-Nearest Neighbors (KNN)'
    features, target, k,test_size ,window_size= attribute_selection(df,option)
    if len(features) > 0 and len(target) > 0:
        y_pred,y_test=Knn_algorithme(df,k,features,target,test_size,window_size)
        st.write(confusion_matrix(y_test, y_pred))
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Precision:", precision_score(y_test, y_pred))
        st.write("Recall:", recall_score(y_test, y_pred))
        st.write("F1 Score:",f1_score(y_test, y_pred))
        st.write("ROC AUC Score:",  roc_auc_score(y_test, y_pred))
        prediction=pd.DataFrame({'y_test':y_test,'y_pred':y_pred})
        fig = px.line(prediction)
        st.plotly_chart(fig)
        # Display the plot in the Streamlit app
        
        
        
def SVM_algorith(df):
    option='Support Vector Machine (SVM) '
    selected_attributes, target_att, Kernel,c,Gamma,test_size,window_size= attribute_selection(df,option)
    st.write(SVM_algorithme( df,selected_attributes,target_att,c,Kernel,Gamma,test_size,window_size))
    
def LSTM(df):
    option='LSTM Long-Short-Term-Memory (LSTM)'
    selected_attributes, target_att, epoch,window_size,test_size= attribute_selection(df,option)
    st.write(LSTM_algorithm(df,epoch,selected_attributes,target_att,test_size,window_size))
  
    


if options == 'K-Nearest Neighbors (KNN) ':
    KNN_algorithme(df)
if options=='LSTM Long-Short-Term-Memory (LSTM)':
    LSTM(df)
    
if options=='Support Vector Machine (SVM) ':
    SVM_algorith(df)