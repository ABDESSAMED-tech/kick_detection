import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import seaborn as sns
import plotly_express as px

def add_variation_column(df, column_name):
    variations = [None]  # Initialize the list of variations with None for the first row
    for i in range(1, len(df)):
        diff = df[column_name].iloc[i] - df[column_name].iloc[i-1]
        variations.append(diff)
    df[f'variation_{column_name}'] = variations
    return df

def preapation_data(df):
    cols=['TVA (m3)', 'SPPA (kPa)', 'MFOP ((m3/s)/(m3/s))',
       'GASA (mol/mol)']
    data=df[cols].copy()
    for column_name in cols:
        data=add_variation_column(data, column_name)
    data=data.dropna()
    data['STATUS']=df['STATUS'][1:]
    data = data.rename(columns={'variation_GASA (mol/mol)': 'variation_GASA'})
    data = data.rename(columns={'variation_SPPA (kPa)':'variation_SPPA'})
    data = data.rename(columns={'variation_MFOP ((m3/s)/(m3/s))': 'variation_MFOP'})
    data = data.rename(columns={'variation_TVA (m3)': 'variation_TVA'})
    return data

def feature_rep(window, thrsh_tva, thrsh_gasa, thrsh_spp, thrsh_mfop):
    # threshold for each feature mean of the feature when the kick is happen
    gasa = window['GASA (mol/mol)'].mean()
    tva = window['variation_TVA'].mean()
    spp = window['variation_SPPA'].mean()
    mfop = window['variation_MFOP'].mean()
    # gasa pas de variasion
    l = []
    if tva > thrsh_tva and mfop < thrsh_mfop and gasa > thrsh_gasa:  # well 19
        l = [int(gasa > thrsh_gasa), int(tva > thrsh_tva),
             int(mfop < thrsh_mfop), int(spp < thrsh_spp)]

        return 1, l
    if gasa > thrsh_gasa and tva > thrsh_tva and spp < thrsh_spp:  # well 15
        l = [int(gasa > thrsh_gasa), int(tva > thrsh_tva),
             int(mfop < thrsh_mfop), int(spp < thrsh_spp)]

        return 1, l
    elif gasa > thrsh_gasa and mfop < thrsh_mfop and spp < thrsh_spp:  # well 6,8
        l = [int(gasa > thrsh_gasa), int(tva > thrsh_tva),
             int(mfop < thrsh_mfop), int(spp < thrsh_spp)]
        return 1,l

    #     return 1, l
    if gasa > thrsh_gasa and tva > thrsh_tva and mfop < thrsh_mfop and spp < thrsh_spp:  # well 26,29
        l = [int(gasa > thrsh_gasa), int(tva > thrsh_tva),
             int(mfop < thrsh_mfop), int(spp < thrsh_spp)]
        return 1, l
    else:

        l = [int(gasa > thrsh_gasa), int(tva > thrsh_tva),
             int(mfop < thrsh_mfop), int(spp < thrsh_spp)]
        return 0, l
    
st.title('kick detection using algorithme approch')

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
                          'detected 0', 'detected 1'])

        st.subheader('Confusion Matrix')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        st.pyplot()
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred))
    st.write("Recall:", recall_score(y_test, y_pred))
    st.write("F1 Score:", f1_score(y_test, y_pred))
    prediction = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    fig = px.line(prediction, color_discrete_sequence=['red', 'blue'])
    fig.update_layout(title='Actual and Detected Values')
    st.plotly_chart(fig)
def alogorthme_approch(df):
    data=preapation_data(df)
    with st.form('approch alogo'):
        window_size=st.slider('Select the window size (second)', min_value=60, max_value=500,step=5,value=300)
        thrsh_tva = st.number_input("Select the threshold of TVA", value=0.01, step=0.001,format="%.000f")
        thrsh_gasa=st.number_input("Select the threshold of GASA ", value=0.01, step=0.001)
        thrsh_spp=st.number_input("Select the threshold of SPP ", value=0.0, step=0.001)
        thrsh_mfop=st.number_input("Select the threshold of MFOP ", value=0.0, step=0.001)
        submit_button=st.form_submit_button('detection')
    if submit_button:
        print('aproch algo')
        data['kick_recognition']=111
        window_size=window_size//5
        st.write(data.shape)
        for i in range(len(data)-window_size+1):
            window = data.iloc[i:i+window_size]
            data['kick_recognition'][i:i+window_size+1],b = feature_rep(
                window, thrsh_tva, thrsh_gasa, thrsh_spp, thrsh_mfop)
        get_metrics1(data['STATUS'], data['kick_recognition'])
        print('end aproch algo')
        




if 'df' in st.session_state:
    df = st.session_state['df']
    alogorthme_approch(df)
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