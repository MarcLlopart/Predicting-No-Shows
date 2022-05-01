import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import plotly.graph_objects as go
import imageio

dataset = pd.read_csv('df_new.csv')
st.set_page_config(
    page_title="Predicting No Show Appointments",
    page_icon="✅",
    layout="wide",
)

st.title("Predicting No Show Appointments")

df = dataset[["num_prior_appointments", "total_cost_public_health", "age", "cost_per_visit"]]
variables = st.selectbox("Select the variable",dataset.columns)
model = st.selectbox("Select the model",['XGB', 'Random Forest', 'Ada Boost'])

fig_col1, fig_col2 = st.columns(2)
with fig_col1:
    st.markdown("### Data distribution by variable")
    fig = px.histogram(dataset, x=variables, nbins=30, range_x=[-0.02,1.14])
    st.write(fig)

rf = pd.read_csv('submissionRF.csv')
xg = pd.read_csv('submissionXGB.csv')
ada = pd.read_csv('submissionADA.csv')
with fig_col2:
    st.markdown("### Predictions by model")
    if model == "XGB":
        df = xg
    elif model == "Random Forest": 
        df = rf
    else:
        df = ada
    fig2 = go.Figure(data=[go.Table(header=dict(values=list(df.columns)),
                 cells=dict(values=df.transpose().values.tolist()))
                     ])
    st.write(fig2)

fig_col3, fig_col4 = st.columns(2)


with fig_col3: 
    st.markdown("### ROC and PR Curves")
    if model == "XGB":
        img = imageio.imread('CurvasXGB.png')
    elif model == "Random Forest": 
        img = imageio.imread('CurvasRF.png')
    else:
        img = imageio.imread('CurvasADA.png')
    fig3 = px.imshow(img, binary_format="png", binary_compression_level=0)
    st.write(fig3)

with fig_col4: 
    st.markdown("### Feature Importance")
    if model == "XGB":
        img = imageio.imread('XGBFeatures.png')
    elif model == "Random Forest": 
        img = imageio.imread('RFFeatures.png')
    else:
        img = imageio.imread('ADAFeatures.png')
    fig4 = px.imshow(img, binary_format="png", binary_compression_level=0)
    st.write(fig4)