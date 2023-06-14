import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,roc_auc_score
import numpy as np
import pandas as pd
import plotly_express as px
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *

st.title("Machine learning algorithme for detection and prediction the kick in drilling operations")




def get_metrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])

    # Plot the confusion matrix using seaborn
    st.subheader('Confusion Matrix')
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    st.pyplot()
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred))
    st.write("Recall:", recall_score(y_test, y_pred))
    st.write("F1 Score:", f1_score(y_test, y_pred))
    # st.write("ROC AUC Score:",  roc_auc_score(y_test, y_pred))
    prediction = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    
    fig = px.line(prediction, color_discrete_sequence=['red', 'blue'])
    fig.update_layout(title='Test and Predicted Values')
    st.plotly_chart(fig)
    

def model_selection(df,option):
    if option=='LSTM Long-Short-Term-Memory (LSTM)':
        with st.form("attribute_form"):
            detect_predict=st.selectbox('Select the option to use the machine learning algorithm for :',options=['Detection','Prediction'])
            selected_attributes = st.multiselect("Select attributes", df.columns)
            target_att = st.selectbox('Select target attribute', df.columns)
            epoch = st.slider('Select number of  epochs', min_value=10, max_value=300)
            batch_size=st.slider('Select number of steps (batch_size)', min_value=16, max_value=256)
            test_size=st.slider('Select the test size', min_value=1, max_value=100,step=1)
            window_size=st.slider('Select the window size', min_value=12, max_value=100,step=1)

            submit_button = st.form_submit_button(label='Run LSTM')
        if submit_button and len(selected_attributes) > 0 and len(target_att):
            if detect_predict=='Detection':  
                history, y_pred, y_test=LSTM_algorithm_detection(df, epoch, batch_size, selected_attributes, target_att, test_size, window_size)
            else:
                history, y_pred, y_test=LSTM_algorithm_prediction(df, epoch, batch_size, selected_attributes, target_att, test_size, window_size)
            get_metrics(y_test,y_pred)
            st.plotly_chart(fig)
            fig = px.line(
                x=range(len(history.history['accuracy'])),
                y=[history.history['accuracy'], history.history['val_accuracy']],
                labels={'x': 'Epochs', 'y': 'Accuracy'},
                color_discrete_sequence=['blue', 'red']
            )

            fig.update_layout(title='Training and Validation Accuracy')

            st.plotly_chart(fig)
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            data = pd.DataFrame({'Epochs': range(len(loss)),
                                'Training Loss': loss,
                                'Validation Loss': val_loss})

            fig = px.line(data, x='Epochs', y=['Training Loss', 'Validation Loss'])
            fig.update_layout(title='Training and Validation Loss')

            st.plotly_chart(fig)
            
        
    if option=='Support Vector Machine (SVM) ':
        with st.form("attribute_form"):
            detect_predict=st.selectbox('Select the option to use the machine learning algorithm for :',options=['Detection','Prediction'])
            selected_attributes = st.multiselect("Select attributes", df.columns)
            target_att = st.selectbox('Select target attribute', df.columns)
            kernel = st.selectbox('Select kernel ', options=['rbf','linear'] )
            window_size=st.slider('Select the window size', min_value=12, max_value=100,step=1)
            c = st.slider('Select a range of Regularization parameter (C)', 0.1, 10.0 )
            gamma= st.slider('Select a range of Paramètre gamma (gamma)', 0.01, 10.0)
            test_size=st.slider('Select the test size', min_value=1, max_value=100,step=1)
            submit_button = st.form_submit_button(label='Run SVM')
        if submit_button and len(selected_attributes) > 0 and len(target_att):
            if detect_predict=='Detection':  
                y_test,y_pred=SVM_algorithme_detection( df,selected_attributes,target_att,c,kernel,gamma,test_size,window_size)
            else:
                 y_test,y_pred=SVM_algorithme_prediction( df,selected_attributes,target_att,c,kernel,gamma,test_size,window_size)

                
            get_metrics(y_test,y_pred)

            
                
    if option=='Random Forest':
        with st.form("attribute_form"):
            detect_predict=st.selectbox('Select the option to use the machine learning algorithm for :',options=['Detection','Prediction'])
            selected_attributes = st.multiselect("Select attributes", df.columns)
            target_att = st.selectbox('Select target attribute', df.columns)
            window_size=st.slider('Select the window size', min_value=12, max_value=100,step=1)
            n_estimators = st.slider('Select  the number of decision trees', 100, 1000,step=50 )
            max_depth= st.slider('Select the maximum depth of each decision tree ', 5, 50,step=5)
            min_samples_split= st.slider(' Select the minimum number of samples required to split an internal node ', 1, 10)

            min_samples_leaf= st.slider('Select the minimum number of samples required to be at a leaf node ',  1, 10)
            test_size=st.slider('Select the test size', min_value=1, max_value=100,step=1)

            submit_button = st.form_submit_button(label='Run Random forest')
        
        if submit_button and len(selected_attributes) > 0 and len(target_att):
            if detect_predict=='Detection': 
                y_test,y_pred=RandomForest_detection(df, selected_attributes, target_att, test_size, window_size, n_estimators, max_depth, min_samples_split, min_samples_leaf)
            else:
                y_test,y_pred=RandomForest_prediction(df, selected_attributes, target_att, test_size, window_size, n_estimators, max_depth, min_samples_split, min_samples_leaf)
            get_metrics(y_test,y_pred)

options = st.radio("Select an algorithme  ", options=[
                   'Support Vector Machine (SVM) ',
                   'LSTM Long-Short-Term-Memory (LSTM)',
                   'Random Forest'
                   ])
if 'df' in st.session_state:
    df = st.session_state['df']

    if options=='Support Vector Machine (SVM) ':
        model_selection(df,options)
    elif options=='LSTM Long-Short-Term-Memory (LSTM)':
        model_selection(df,options)
    else :
        model_selection(df,options)
else:
    st.warning('Please load the dataset !!')
    
        
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

# Render the footer using the st.beta_container() function
st.container().write(footer, unsafe_allow_html=True)