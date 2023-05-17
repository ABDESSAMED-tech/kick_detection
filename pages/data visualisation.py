import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly_express as px
import seaborn as sns
import streamlit_vega_lite as st_vl
# import seaborn as sns
st.title('Data visualisation ')
df = st.session_state['df']


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


def line_plot(df):
    x_axis = st.selectbox('Select the X-Axis Value', options=df.columns)
    col = st.color_picker('Select a plot color')
    plot = px.line(df, x_axis, color=df['STATUS'])
    plot.update_traces(marker=dict(color=col))
    st.plotly_chart(plot)
    # st.write(df['STATUS'] == 1)


def correlation_matrix(df):
    st.header("matrix of correlation")
    selected_attrebut, status = attribute_selection_corr(df)
    if selected_attrebut and status:
        corr_matrix = df[selected_attrebut][df['STATUS'] == 1].corr()
        heatmap = (
            corr_matrix.style.background_gradient(cmap="coolwarm")
            .set_precision(2)
            .set_properties(**{"font-size": "12pt", "width": "50px", "text-align": "center"})
            .set_caption("Correlation Matrix")
        )
        st.write(heatmap)
    else:
        corr_matrix = df[selected_attrebut].corr()
        heatmap = (
            corr_matrix.style.background_gradient(cmap="coolwarm")
            .set_precision(2)
            .set_properties(**{"font-size": "12pt", "width": "50px", "text-align": "center"})
            .set_caption("Correlation Matrix")
        )
        st.write(heatmap)


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
        g.map_diag(plt.hist)
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
            "Select the Y-Axis Values", df.columns)
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


options = st.radio("Type of visualisation ", options=[
                   'Table visualisation',
                   'Scartter plot visualisation',
                   'Scartter plot with Intervalle visualisation',
                   'Line plot visualisation',
                   'Line plot with Intervalle visualisation', 'Histgoramme',
                   'Box plot', 'Show all graphs',
                   'Multiple attribute', 'matrix of correlation'])
if options == 'Table visualisation':
    # table_vis(df)
    pass
elif options == 'matrix of correlation':
    correlation_matrix(df)
elif options == 'Line plot visualisation':
    line_plot(df)

elif options == 'Box plot':
    box_plot(df)
elif options == 'Line plot with Intervalle visualisation':
    line_plot_Inte_vis(df)
elif options == 'Scartter plot with Intervalle visualisation':
    plot_Intervale(df)
elif options == 'Show all graphs':
    paire_plot(df)
elif options == 'Histgoramme':
    histograme(df)
elif options == 'Multiple attribute':
    plot_line(df)
else:
    interactive_plt(df)
