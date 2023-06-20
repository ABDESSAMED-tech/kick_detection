import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import pandas as pd
import plotly_express as px
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile
import os
from utils import *
import pickle
import json

# df = st.session_state['df']
st.set_option('deprecation.showPyplotGlobalUse', False)


options = st.sidebar.radio("Select an algorithme for detection  ", options=[
                   'Support Vector Machine (SVM) ',
                   'LSTM Long-Short-Term-Memory (LSTM)',
                   'Random Forest'
                   ])
st.title(str(options))
st.header('Saved models')


def select_range(df, step):
    start, end = st.slider(
        "Select a range of data",
        min_value=0,
        max_value=len(df) - 1,
        value=(0, len(df) - 1),
        step=step
    )
    return start, end


def get_metrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=[
                         'Predicted 0', 'Predicted 1'])

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


def get_metrics1(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    # Check if the confusion matrix contains only zeros
    if len(cm) < 2:
        st.subheader('Confusion Matrix')
        st.write(cm)
        # cm_df = pd.DataFrame([cm[0]], index=['Actual 0'],
        #                      columns='Predicted 0')
        # st.subheader('Confusion Matrix')
        # sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')

    else:
        cm = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=[
                          'Predicted 0', 'Predicted 1'])

        st.subheader('Confusion Matrix')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        st.pyplot()
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred))
    st.write("Recall:", recall_score(y_test, y_pred))
    st.write("F1 Score:", f1_score(y_test, y_pred))
    prediction = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    fig = px.line(prediction, color_discrete_sequence=['red', 'blue'])
    fig.update_layout(title='Test and Predicted Values')
    st.plotly_chart(fig)


def load_models_prediction(X, option, model_file):
    
    # Save the uploaded file to a temporary location
    temp_model_file = tempfile.NamedTemporaryFile(delete=False)
    temp_model_file.write(model_file.read())
    temp_model_file.close()
    if option == 'Support Vector Machine (SVM) ' or option=='Random Forest':
        # st.write(option)
        with open(temp_model_file.name, "rb") as f:
            model = pickle.load(f)
            # Make a prediction.
        prediction = model.predict(X)

        # Print the prediction.
        return prediction
    else:
        model = tf.keras.models.load_model(temp_model_file.name)
        prediction = model.predict(X)
        y_pred = (prediction >= 0.5).astype(int)
        return y_pred


def prepare_data_svm_random_forest(df, Features, target, p, window, test_size):
    # st.write(test_size/100)
    # st.write(window//5)
    save_path = r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\Save Models\normalization_params_LSTM.json'
    with open(save_path, 'r') as f:
        normalization_params = json.load(f)
    loaded_min = np.array(normalization_params['min'])
    loaded_max = np.array(normalization_params['max'])
    X = df[Features].values
    y = df[target].values
    X = (X - loaded_min) / (loaded_max - loaded_min)
    if p == 'Detection':
        segments, labels = segmantation(X, y, window_length=window//5)
        _, X_test, _, y_test = split_data_balanced(
            segments, labels, test_size=test_size/100)
        # Reshape the feature matrices for SVM
        X_test = X_test.reshape(X_test.shape[0], -1)
    else:
        segments, labels = segmentation_prediction(X, y, window//5, 0.2)
        _, X_test, _, y_test = split_data_balanced(
            segments, labels, test_size=test_size/100)
        # Reshape the feature matrices for SVM
        X_test = X_test.reshape(X_test.shape[0], -1)
    return X_test, y_test


def prepare_data_LSTM(df, Features, target, p, window_size, test_size):
    save_path = r'C:\Users\hp\Desktop\M2\PFE\Code\code pfe\Coud source\Code\Save Models\normalization_params_LSTM.json'
    with open(save_path, 'r') as f:
        normalization_params = json.load(f)
    loaded_min = np.array(normalization_params['min'])
    loaded_max = np.array(normalization_params['max'])
    X = df[Features].values
    y = df[target].values
    X = (X - loaded_min) / (loaded_max - loaded_min)
    if p == 'Detection':
        segments, labels = segmantation(X, y, window_size//5)
    else:
        segments, labels = segmentation_prediction(X, y, window_size//5, 0.2)

    _, X_test, _, y_test = split_data_balanced(
        segments, labels, test_size=test_size/100)
    X_test, y_test = np.array(X_test), np.array(y_test).reshape(- 1, 1)
    # st.write(f'x.shape {X_test.shape}, \n ytest.shape: {y_test.shape}')
    return X_test, y_test


st.set_option('deprecation.showfileUploaderEncoding', False)

COLS=[ 'TVA (m3)', 'SPPA (kPa)', 'MFOP ((m3/s)/(m3/s))', 'GASA (mol/mol)']
target=['STATUS']
def model_selection(df, option):
    if 'X_test' not in st.session_state:
        st.session_state['X_test'] = []
    if 'y_test' not in st.session_state:
        st.session_state['y_test'] = []
    with st.form('window_test', clear_on_submit=False):
        window_size = st.slider(
            'Select the window size(second)', min_value=60, max_value=500, step=60, value=300)
        test_size = st.slider('Select the test size',
                              min_value=1, max_value=40, step=1, value=20)
        selected_attributes = COLS
        target_att = target
        
        detect_predict = st.selectbox(
            'Select the option to use the machine learning algorithm for :', options=['Detection', 'Prediction'])
        if option=='LSTM Long-Short-Term-Memory (LSTM)': 
            model_file = st.file_uploader("Upload Model", type=".h5")
        else:
            model_file = st.file_uploader("Upload Model", type=".pkl")

        submit_button_test = st.form_submit_button(label='Get test data ')

    if submit_button_test and len(selected_attributes) > 0 and len(target_att):
        if option=='Support Vector Machine (SVM) ' or option=='Random Forest':
            st.session_state['X_test'], st.session_state['y_test'] = prepare_data_svm_random_forest(
                df, selected_attributes, target_att, detect_predict, window_size, test_size)
        else:
             st.session_state['X_test'], st.session_state['y_test'] = prepare_data_LSTM(
                df, selected_attributes, target_att, detect_predict, window_size, test_size)
    
          
    with st.form('run prediction'):
            if len(st.session_state['X_test'])>0:
                start, end = select_range(st.session_state['X_test'], window_size//5)
            run_button_clicked = st.form_submit_button(
                    str(detect_predict)+' with '+str(option))

    if run_button_clicked and model_file :
            y_pred = load_models_prediction(
                st.session_state['X_test'][start:end], option, model_file)
            if option=='LSTM Long-Short-Term-Memory (LSTM)':
                    st.session_state['y_test'], y_pred=st.session_state['y_test'].reshape(st.session_state['y_test'].shape[0],), y_pred.reshape(y_pred.shape[0])
            
            get_metrics1(st.session_state['y_test'][start:end], y_pred)

    else:
        st.warning("Please Load the saved model !!")

if 'df' in st.session_state:
    df = st.session_state['df']
    # cols=['TVA (m3)', 'SPPA (kPa)', 'MFOP ((m3/s)/(m3/s))',
    #    'GASA (mol/mol)']
    # X = df[[ 'TVA (m3)', 'SPPA (kPa)', 'MFOP ((m3/s)/(m3/s))', 'GASA (mol/mol)']].values
    # y = df['STATUS'].values
    # _, X_test, _, y_test = split_data_balanced(X, y, test_size=0.2)
    # data=pd.DataFrame(X_test,columns=cols)
    # data['STATUS']=y_test

    model_selection(df, options)
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

# Render the footer using the st.beta_container() function
st.container().write(footer, unsafe_allow_html=True)
