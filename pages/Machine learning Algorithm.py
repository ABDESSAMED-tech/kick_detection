import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,roc_auc_score
import numpy as np
import pandas as pd
import plotly_express as px
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
import pickle
from io import BytesIO
import base64
options = st.sidebar.radio("Select an algorithme  ", options=[
                   'Support Vector Machine (SVM) ',
                   'LSTM Long-Short-Term-Memory (LSTM)',
                   'Random Forest'
                   ])
st.title(str(options))

st.set_option('deprecation.showPyplotGlobalUse', False)


def download_model_lstm(name, recall, precision, fscore,window, model):
    precision_formatted = round(precision, 3)
    recall_formatted = round(recall, 3)
    fscore_formatted = round(fscore, 3)
    model_filename = f"{name}_model_recall_{recall_formatted}_precision_{precision_formatted}_f_measure_{fscore_formatted}_window_{window}.h5"

    # Save the LSTM model
    model.save(model_filename)

    # Read the saved model file
    with open(model_filename, "rb") as f:
        model_bytes = f.read()

    # Encode the model bytes to base64 and create the download link
    model_base64 = base64.b64encode(model_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{model_base64}" download="{model_filename}">Download Model</a>'
    return href
    
def download_model(name,recall,precision,fscore,window,model):
    precision_formatted = round(precision, 3)
    recall_formatted = round(recall, 3)
    fscore_formatted = round(fscore, 3)
    model_filename = f"{name}_model_recall_{recall_formatted},_precion_{precision_formatted}_f_mesure{fscore_formatted}_window_{window}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    with open(model_filename, "rb") as f:
        model_bytes = f.read()
    href = f'<a href="data:application/octet-stream;base64,{base64.b64encode(model_bytes).decode()}" download="{model_filename}">Download Model</a>'
    return href

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

COLS=[ 'TVA (m3)', 'SPPA (kPa)', 'MFOP ((m3/s)/(m3/s))', 'GASA (mol/mol)']
target=['STATUS']
def model_selection(df,option):
    if option=='LSTM Long-Short-Term-Memory (LSTM)':
        with st.form("attribute_form"):
            detect_predict=st.selectbox('Select the option to use the machine learning algorithm for :',options=['Detection','Prediction'])
            selected_attributes = COLS
            target_att = target
            epoch = st.slider('Select number of  epochs', min_value=10, max_value=300,value=100)
            batch_size=st.slider('Select number of steps (batch_size)', min_value=16, max_value=256,value=32)
            test_size=st.slider('Select the test size', min_value=1, max_value=40,step=1,value=20)
            window_size=st.slider('Select the window size (second)', min_value=60, max_value=500,step=5,value=300)

            submit_button = st.form_submit_button(label='Run LSTM')
        if submit_button and len(selected_attributes) > 0 and len(target_att):
            if detect_predict=='Detection':  
                model,history, y_pred, y_test=LSTM_algorithm_detection(df.iloc[14000:15000], epoch, batch_size, selected_attributes, target_att, test_size, window_size//5)
            else:
                model,history, y_pred, y_test=LSTM_algorithm_prediction(df, epoch, batch_size, selected_attributes, target_att, test_size, window_size//5)
            y_test, y_pred=y_test.reshape(y_test.shape[0],), y_pred.reshape(y_pred.shape[0])
            get_metrics(y_test,y_pred)
            
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
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            fscore=f1_score(y_test, y_pred)
            st.subheader("Download Trained SVM Model")
            st.markdown(download_model_lstm('LSTM_'+str(detect_predict)+'_', recall, precision, fscore,window_size, model), unsafe_allow_html=True)        
             
            
            
            
            
            
        
    if option=='Support Vector Machine (SVM) ':
        with st.form("attribute_form"):
            detect_predict=st.selectbox('Select the option to use the machine learning algorithm for :',options=['Detection','Prediction'])
            selected_attributes = COLS
            target_att = target
            kernel = st.selectbox('Select kernel ', options=['rbf','linear'] )
            window_size=st.slider('Select the window size (second)', min_value=60, max_value=500,step=1,value=300)
            c = st.slider('Select a range of Regularization parameter (C)', 0.1, 10.0,value=10.0 )
            gamma= st.slider('Select a range of Paramètre gamma (gamma)',min_value= 0.01,max_value= 10.0,value=1.0,step=0.01)
            test_size=st.slider('Select the test size', min_value=1, max_value=100,step=1,value=20)
            submit_button = st.form_submit_button(label='Run SVM')
        if submit_button and len(selected_attributes) > 0 and len(target_att):
            if detect_predict=='Detection':  
                model,y_test,y_pred=SVM_algorithme_detection( df,selected_attributes,target_att,c,kernel,gamma,test_size,window_size//5)
            else:
                model,y_test,y_pred=SVM_algorithme_prediction( df,selected_attributes,target_att,c,kernel,gamma,test_size,window_size//5)

               
            get_metrics(y_test,y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            fscore=f1_score(y_test, y_pred)
            st.subheader("Download Trained SVM Model")
            st.markdown(download_model('SVM_'+str(detect_predict)+'_',recall,precision,fscore,window_size,model), unsafe_allow_html=True)        
                
    if option=='Random Forest':
        with st.form("attribute_form"):
            detect_predict=st.selectbox('Select the option to use the machine learning algorithm for :',options=['Detection','Prediction'])
            selected_attributes = COLS
            target_att = target
            window_size=st.slider('Select the window size(second)', min_value=60, max_value=500,step=60,value=300)
            n_estimators = st.slider('Select  the number of decision trees', 100, 1000,step=50,value=200 )
            max_depth= st.slider('Select the maximum depth of each decision tree ', 5, 50,step=1,value=10)
            min_samples_split= st.slider(' Select the minimum number of samples required to split an internal node ', 1, 10,value=2)

            min_samples_leaf= st.slider('Select the minimum number of samples required to be at a leaf node ',  1, 10)
            test_size=st.slider('Select the test size', min_value=1, max_value=100,step=1,value=20)

            submit_button = st.form_submit_button(label='Run Random forest')
        
        if submit_button and len(selected_attributes) > 0 and len(target_att):
            if detect_predict=='Detection': 
                model,y_test,y_pred=RandomForest_detection(df, selected_attributes, target_att, test_size, window_size, n_estimators, max_depth, min_samples_split, min_samples_leaf)
            else:
                model, y_test,y_pred=RandomForest_prediction(df, selected_attributes, target_att, test_size, window_size, n_estimators, max_depth, min_samples_split, min_samples_leaf)
            get_metrics(y_test,y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            fscore=f1_score(y_test, y_pred)
           
            st.markdown(download_model('Random forest_'+str(detect_predict)+'_',recall,precision,fscore,window_size,model), unsafe_allow_html=True)        
                


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
    <p>This app is developed by BOULARIACHE Abdessamed and TAZIR Mohamed Reda.</p>
</div>
'''

# Render the footer using the st.beta_container() function
st.container().write(footer, unsafe_allow_html=True)