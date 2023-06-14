import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import pandas as pd
import plotly_express as px
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
st.title("Machine learning algorithme for detection and prediction the kick in drilling operations")
st.header('Best models')
# df = st.session_state['df']
st.set_option('deprecation.showPyplotGlobalUse', False)


options = st.radio("Select an algorithme for detection  ", options=[
                   'Support Vector Machine (SVM) ',
                   'LSTM Long-Short-Term-Memory (LSTM)',
                   'Random Forest'
                   ])


def select_range(df):
    start, end = st.slider(
        "Select a range of data",
        min_value=0,
        max_value=len(df) - 1,
        value=(0, len(df) - 1),
        step=60
    )
    return start, end

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


def model_selection(df, option):

    with st.form("attribute_form"):
        st.write('You select :', option)
        start, end = select_range(df)
        st.write('Start Test:', start)
        st.write('End Test:', end)
        detect_predict = st.selectbox(
            'Select the option to use the machine learning algorithm for :', options=['Detection', 'Prediction'])
        selected_attributes = st.multiselect("Select attributes", df.columns)
        target_att = st.selectbox('Select target attribute', df.columns)
        submit_button = st.form_submit_button(label='Run '+str(option))
    if submit_button and len(selected_attributes) > 0 and len(target_att):
      
        y_test, y_pred = load_models_prediction(
            df.iloc[start:end], selected_attributes, target_att, option, detect_predict)
        if option=='LSTM Long-Short-Term-Memory (LSTM)':
            y_test, y_pred=y_test.reshape(y_test.shape[0],), y_pred.reshape(y_pred.shape[0],)
            st.write('y_test:',y_test.shape)
            st.write('y_pred',y_pred.shape)
        get_metrics(y_test, y_pred)


if 'df' in st.session_state:
    df = st.session_state['df']

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
    <p>This app is developed by BOULARIACHE Abdessamed and TAZIR Mouhamed Reda.</p>
</div>
'''

# Render the footer using the st.beta_container() function
st.container().write(footer, unsafe_allow_html=True)

# Render the footer using the st.beta_container() function
st.container().write(footer, unsafe_allow_html=True)