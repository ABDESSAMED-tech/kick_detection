import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, mean_absolute_error, matthews_corrcoef, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import cross_val_predict
import numpy as np
import pandas as pd
import plotly_express as px 
import seaborn as sns
import matplotlib.pyplot as plt
df = st.session_state['df'] 
st.title("Machine Learning Algorithme")


options = st.radio("Select an algorithme  ", options=[
                   'K-Nearest Neighbors (KNN) ',
                   'Multi layer perceptron (MLP)',
                   'LSTM Long-Short-Term-Memory (LSTM)',
                   ])
def attribute_selection(df):
    
    with st.form("attribute_form"):
        selected_attributes = st.multiselect("Select attributes", df.columns)
        target_att=st.selectbox('Select target attribute',df.columns)
        k=st.slider('Select value of K',min_value=2,max_value=100)
        submit_button = st.form_submit_button(label='Run')
    
    # Return the selected attributes
    return selected_attributes,target_att,k
def knn(df):
    num_folds = 10
    
    selected_attributes ,target,k= attribute_selection(df)
    if len(selected_attributes)>0 and len(target)>0:
        X=df[selected_attributes]
        y=df [target]
        knn = KNeighborsClassifier(n_neighbors=k)
        y_pred = cross_val_predict(knn, X, y, cv=num_folds)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        mae = mean_absolute_error(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        st.write('Accuracy:', accuracy)
        st.write('Precision:', precision)
        st.write('Recall:', recall)
        st.write('F1 score:', f1)
        st.write('MAE:', mae)
        st.write('MCC:', mcc)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"], ax=ax) 
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)
        
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)
        st.write('Roc_auc= {}, threshods :{} '.format(roc_auc,thresholds))

        roc_data = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr})

        fig1 = px.line(roc_data, x='False Positive Rate', y='True Positive Rate', title='Receiver Operating Characteristic')
        fig1.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        fig1.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', showlegend=False)
        st.plotly_chart(fig1)
        precision, recall, thresholds = precision_recall_curve(y, y_pred)
        prc_auc = auc(recall, precision)
        st.write('prc_auc= {} '.format(prc_auc,))

        prc_data = pd.DataFrame({'Precision': precision, 'Recall': recall})

        fig2 = px.line(prc_data, x='Recall', y='Precision', title='Precision-Recall Curve')
        fig2.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=1, y1=0)
        fig2.update_layout(xaxis_title='Recall', yaxis_title='Precision', showlegend=False)
        st.plotly_chart(fig2)
    
    
if options=='K-Nearest Neighbors (KNN) ':
    knn(df)