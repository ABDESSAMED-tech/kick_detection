import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly_express as px
import seaborn as sns
import streamlit_vega_lite as st_vl
import numpy as np
# import seaborn as sns
st.set_page_config(page_title="Data Visualisation", page_icon='📊')

st.sidebar.title('Data visualisation ')
# df = st.session_state['df']


def box_plot(df):
    st.header('Box Plot')
    option = st.selectbox('Select the attribute  ', options=df.columns)
    if option != 'index':
        plot = px.box(df, x=df[option])
        st.plotly_chart(plot)


def interactive_plt(df):
    x_axis = st.selectbox('Select the X-Axis Value', options=df.columns)
    y_axis = st.selectbox('Select the Y-Axis Value', options=df.columns)
    # col = st.color_picker('Select a plot color')

    plot = px.scatter(df, x=x_axis, y=y_axis,
                      color=df['STATUS'], color_continuous_scale=['blue', 'red'])
    # plot.update_traces(marker=dict(color=col))
    st.plotly_chart(plot)


def table_vis(df):
    st.table(df)


# def line_plot(df):
#     x_axis = st.selectbox('Select the X-Axis Value', options=df.columns)
#     col = st.color_picker('Select a plot color')
#     plot = px.line(df, x_axis, color=df['STATUS'])
#     plot.update_traces(marker=dict(color=col))
#     st.plotly_chart(plot)
#     # st.write(df['STATUS'] == 1)
def  select_subplot(df):
        with st.form("attribute_form"):
            selected_attrs = st.multiselect(
                "Select the columns", df.columns)
            
            submit_button = st.form_submit_button(label='Run')
        return selected_attrs
def subplot(df):
    
    selected_attrs=select_subplot(df)

            # Iterate over each selected attribute
    if len(selected_attrs)>0:
        fig, axes = plt.subplots(len(selected_attrs), 1, figsize=(8, 4 * len(selected_attrs)))

        for i, attr in enumerate(selected_attrs):
                    ax = axes[i] if len(selected_attrs) > 1 else axes  # Select the appropriate axis
                    ax.plot( df[attr])
                    ax.set_ylabel(attr)

                # Adjust spacing between subplots
        plt.tight_layout()

                # Display the subplots using Streamlit
        st.pyplot(fig)
    else:
        st.warning('Please select attribute to plot !!')
    
def correlation_matrix(df):
    st.header("Correlation Matrix")
    selected_attrebut, status = attribute_selection_corr(df)
    if selected_attrebut :
            if status:
                corr_matrix = df[selected_attrebut][df['STATUS'] == 1].corr()
            
                # Create a heatmap using seaborn
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr_matrix, annot=True, cmap="Blues", fmt=".2f", ax=ax)

                # Set plot properties
                plt.title("Correlation Matrix")
                plt.xticks(np.arange(0.5, len(corr_matrix.columns) + 0.5), corr_matrix.columns)
                plt.yticks(np.arange(0.5, len(corr_matrix.columns) + 0.5), corr_matrix.columns)
                plt.tight_layout()

                # Display the plot in Streamlit
                st.pyplot(fig)
            else:
                corr_matrix = df[selected_attrebut].corr()
                # Create a heatmap using seaborn
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr_matrix, annot=True, cmap="Blues", fmt=".2f", ax=ax)

                # Set plot properties
                plt.title("Correlation Matrix")
                plt.xticks(np.arange(0.5, len(corr_matrix.columns) + 0.5), corr_matrix.columns)
                plt.yticks(np.arange(0.5, len(corr_matrix.columns) + 0.5), corr_matrix.columns)
                plt.tight_layout()

                # Display the plot in Streamlit
                st.pyplot(fig)
    else:
            st.warning('Please select attribute')

def show_graph_all_Kick(df):
    x_axis = st.selectbox('Select the X-Axis Value', options=df.columns)
    for i in df:
        if i not in [x_axis, df.columns[df.columns.get_loc('STATUS')+1]]:
            y_axis = df[i]
            plot = px.scatter(
                df, x=x_axis, y=y_axis, color=df['STATUS'], color_continuous_scale=['blue', 'red'])
            # plot.update_traces(marker=dict(colo))
            st.plotly_chart(plot)


def histograme(df):
    # data=df[['TVA (m3)', 'SPPA (kPa)', 'MFOA (m3/s)',
    #    'MFOP ((m3/s)/(m3/s))', 'GASA (mol/mol)']]
    status = st.checkbox('Status==1?')
    option = st.selectbox('Select attribute ', df.columns)
    if status and option:
        hist = sns.histplot(df[option][df['STATUS'] == 1], kde=True)

        st.pyplot(hist.figure)
    elif option:
        hist = sns.histplot(df[option], kde=True)
        st.pyplot(hist.figure)
def paire_plot(df):
    option,status=attribute_selection_corr(df)
    df["color"] = df["STATUS"].apply(lambda x: "red" if x == 1 else "blue")
    if option and status:
        g = sns.PairGrid(data=df[option])
        g.map_diag(sns.histplot)
        g.map_upper(sns.scatterplot,c=df['color'])
        g.map_lower(sns.scatterplot,c=df['color'])
        st.pyplot(g.figure)
        
    elif option:
        g = sns.PairGrid(df[option])
        g.map_diag(plt.hist)
        g.map_upper(sns.scatterplot)
        g.map_lower(sns.scatterplot)
        st.pyplot(g.fig)
        

def plot_func(x, y, **kwargs):
    color = kwargs.pop("color")
    plt.scatter(x, y, c=color, **kwargs) 
    
def attribute_selection_corr(df):

    with st.form("attribute_form"):
        selected_attributes = st.multiselect(
            "Select attributes", df.columns)
        status = st.checkbox('where STATUS =1')
        submit_button = st.form_submit_button(label='Run')
    return selected_attributes, status


def attribute_selection(df):

    with st.form("attribute_form"):
        selected_attributes = st.multiselect(
            "Select the Y-Axis Values", df.columns)
        target_att = st.selectbox('Select the X-Axis Value', df.columns)
        submit_button = st.form_submit_button(label='Run')

    # Return the selected attributes
    return selected_attributes, target_att


def plot_line(df):
    st.header('Multipl attributs')
    selected_attrebut, target_att = attribute_selection(df)
    # print(selected_attrebut)
    fig = px.line(data_frame=df, x=target_att, y=selected_attrebut)
    st.plotly_chart(fig)


def plot_Intervale(df):
    x_axis = st.selectbox('Select the X-Axis Value', options=df.columns)
    y_axis = st.selectbox('Select the Y-Axis Value', options=df.columns)
    # col=st.color_picker('Select a plot color')
    interval1 = st.number_input(
        "Select Interval to start ", min_value=0, max_value=len(df), step=1)
    interval2 = st.number_input(
        "Select Interval to end  ", min_value=10, max_value=len(df), step=1)

    plot = px.scatter(df, df[x_axis][interval1:interval2], df[y_axis][interval1:interval2],
                      color=df['STATUS'][interval1:interval2], color_continuous_scale=['blue', 'red'])
    plot.update_xaxes(title=x_axis)
    plot.update_yaxes(title=y_axis)
    plot.update()
    st.plotly_chart(plot)


def line_plot_Inte_vis(df):
    x_axis = st.selectbox('Select the X-Axis Value', options=df.columns)
    y_axis = st.selectbox('Select the Y-Axis Value', options=df.columns)
    # col=st.color_picker('Select a plot color')
    interval1 = st.number_input(
        "Select Interval to start ", min_value=0, max_value=len(df), step=1)
    interval2 = st.number_input(
        "Select Interval to end  ", min_value=10, max_value=len(df), step=1)
    plot = px.line(df, df[x_axis][interval1:interval2], df[y_axis][interval1:interval2],
                   color=df['STATUS'][interval1:interval2], color_discrete_sequence=['red', 'blue'])
    plot.update()
    plot.update_xaxes(title=x_axis)
    plot.update_yaxes(title=y_axis)
    st.plotly_chart(plot)


options = st.sidebar.radio("Type of visualisation ", options=[
                   'Table visualisation',
                   'Scartter plot visualisation',
                   'Scartter plot with Intervalle visualisation',
                   'Box plot', 'Paire Plot',
                   'Multiple attributs', 'Correlation Matrix','Subplot visualisation'])

if 'df' in st.session_state:
    df = st.session_state['df']
    if options == 'Table visualisation':
        # table_vis(df)
        ...
        
    elif options=='Subplot visualisation':
        subplot(df)
    elif options == 'Correlation Matrix':
        correlation_matrix(df)

    elif options == 'Box plot':
        box_plot(df)
    elif options == 'Scartter plot with Intervalle visualisation':
        plot_Intervale(df)
    elif options == 'Paire Plot':
        paire_plot(df)
    elif options == 'Histgoramme':
        histograme(df)
    elif options == 'Multiple attributs':
        plot_line(df)
    else:
        interactive_plt(df)

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