import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv('./data/raw/Churn_Modelling.csv')

@st.cache
def load_data():

    data = pd.read_csv('./data/raw/Churn_Modelling.csv')
    # lowercase everything.
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    # drop things we don't need.
    data.drop(['rownumber','customerid', 'surname'], axis=1,  inplace=True)
    # binary classification without calling scikit.
    data['gender'] = np.where(data['gender'] == 'Male', 1, 0)

    # custom function for label encoding, again without calling scikit.

    # original

    # geo_label = { ni: n for n,ni in enumerate(set(df['Geography']))}
    # df['Geography'] = df['Geography'].map(geo_label)
    # df['Geography'].value_counts().to_dict())

    # tweaked for testing.
    data['geography'] = pd.Categorical(data['geography'])
    data_dummies = pd.get_dummies(data['geography'], prefix='c')
    data = pd.concat([data,data_dummies],axis=1)

    data.rename(columns={'geography': 'country'}, inplace=True)
    # data.to_csv('data_two.csv')
    return data


def df_info(df=df):

    buffer = io.StringIO()

    df.info(buf=buffer)

    s = buffer.getvalue()

    x = s.split('\n')

    list_1 = []

    for i in x[3:-3]:
        str_list = []
        for c in i.split(' '):
            if c != '':
                str_list.append(c)
        list_1.append(str_list)

    df_info = pd.DataFrame(list_1)
    # we drop the first two columns.
    df_info.drop([0,3], axis=1, inplace=True)
    # we drop the first two rows.
    df_info.drop([0,1], axis=0, inplace=True)

    df_info.columns = ['Variable', '# of Non-Null Values', 'Data Type']

    df_info['# of Non-Null Values'] = df_info['# of Non-Null Values'].astype('int')

    nunique_list = []

    for i in df_info['Variable']:
        nunique_list.append(df[i].nunique())

    df_info['# of Unique Values'] = nunique_list

    df_info = df_info.sort_values(by='# of Unique Values', ascending=False)

    return df_info

def filter_by_dtype(dataframe, data_type):
    """filter a dataframe by columns with a certain data_type"""
    col_names = dataframe.dtypes[dataframe.dtypes == data_type].index
    return dataframe[col_names]

def highlight_cols(s):

    color = 'lightgreen'
    return 'background-color: %s' % color

def param_search():

    infile = open('./para_df.pickle','rb')

    para_df = pickle.load(infile)

    return para_df

def show_histogram_plot(selected_species_df: pd.DataFrame):
    """## Component to show a histogram of the selected species and a selected feature

    Arguments:
        selected_species_df {pd.DataFrame} -- A DataFrame with the same columns as the
            source_df iris dataframe
    """
    # st.subheader("Histogram")

    col = ['exited', 'country', 'gender', 'hascrcard', 'isactivemember', 'numofproducts']
    col_e = ['exited']
    col_p = ['age',  'balance', 'country', 'creditscore', 'estimatedsalary', 'exited', 'gender','hascrcard', 'isactivemember', 'numofproducts', 'tenure']

    feature = st.sidebar.selectbox("Which feature?", col_p)
    feature_two = st.sidebar.selectbox("Which selected feature?", col_e)

    fig2 = px.histogram(selected_species_df, x=feature, color=feature_two, marginal='violin')
    fig2.update_traces(opacity=0.75)

    fig2.update_layout(yaxis_title="", margin=dict(l=0, r=110, t=20, b=20),
        legend=dict(
        x=0,
        y=-0.35,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
            ),
        bgcolor="White",
        bordercolor="White",
        borderwidth=2
        )
    )

    st.plotly_chart(fig2)

def train_models():

    infile = open('./scores_df.pickle', 'rb')

    scores_df = pickle.load(infile)

    return scores_df
