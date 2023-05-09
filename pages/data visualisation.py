import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly_express as px
# import seaborn as sns
st.title('Data visualisation ')
df = st.session_state['df']


def box_plot(df):
    st.header('Box Plot')
    option = st.selectbox('Select the attribute  ', options=df.columns)
    if option!='index'  :  
        plot = px.box(df, x=df[option])
        st.plotly_chart(plot)


def interactive_plt(df):
    x_axis = st.selectbox('Select the X-Axis Value', options=df.columns)
    y_axis = st.selectbox('Select the Y-Axis Value', options=df.columns)
    # col = st.color_picker('Select a plot color')

    plot = px.scatter(df, x=x_axis, y=y_axis,  color=df['STATUS'],color_continuous_scale=['blue','red'])
    # plot.update_traces(marker=dict(color=col))
    st.plotly_chart(plot)


def table_vis(df):
    st.table(df)


def line_plot(df):
    x_axis = st.selectbox('Select the X-Axis Value', options=df.columns)
    col = st.color_picker('Select a plot color')
    plot = px.line(df, x_axis, color=df['STATUS'])
    plot.update_traces(marker=dict(color=col))
    st.plotly_chart(plot)
    # st.write(df['STATUS'] == 1)


def show_graph_all_Kick(df):
    x_axis = st.selectbox('Select the X-Axis Value', options=df.columns)
    for i in df:
        if i not in [x_axis,df.columns[df.columns.get_loc('STATUS')+1]]:
            y_axis=df[i]
            plot = px.scatter(df, x=x_axis, y=y_axis, color=df['STATUS'],color_continuous_scale=['blue','red'])
            # plot.update_traces(marker=dict(colo))
            st.plotly_chart(plot)
            

def histograme(df):
    # data=df[['TVA (m3)', 'SPPA (kPa)', 'MFOA (m3/s)',
    #    'MFOP ((m3/s)/(m3/s))', 'GASA (mol/mol)']]
    for feature in df.columns:
        fig = px.histogram(df, x=feature, nbins=50, title=f"Histogram of {feature}")
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


options = st.radio("Type of visualisation ", options=[
                   'Table visualisation',
                   'Scartter plot visualisation',
                   'Scartter plot with Intervalle visualisation',
                   'Line plot visualisation',
                   'Line plot with Intervalle visualisation','Histgoramme',
                   'Box plot','Show all graphs'])
if options == 'Table visualisation':
    # table_vis(df)
    pass
elif options == 'Line plot visualisation':
    line_plot(df)

elif options == 'Box plot':
    box_plot(df)
elif options ==  'Line plot with Intervalle visualisation':
    line_plot_Inte_vis(df)
elif options == 'Scartter plot with Intervalle visualisation':
    plot_Intervale(df)
elif options=='Show all graphs':
    show_graph_all_Kick(df)
elif options=='Histgoramme':
    histograme(df)
else:
    interactive_plt(df)
