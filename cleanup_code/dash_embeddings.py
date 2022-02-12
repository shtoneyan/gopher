#!/usr/bin/env python
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import wandb
import tensorflow as tf
import h5py
import numpy as np
import plotly.express as px
import flask
import scipy.stats
import plotly.graph_objects as go
from modelzoo import *
import pickle
import custom_fit
import util
import os

# y axis fixed somewhere in the Average
# remove mse
# dropdown menu for cell lines
# plot extra cell lines

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)

def scipy_pr(y_true, y_pred):

    pr = scipy.stats.pearsonr(y_true, y_pred)[0]
    return pr

def np_poiss(y_true, y_pred):
    return y_pred - y_true * np.log(y_pred)

hf = h5py.File('app_data.h5', 'r')
truth = hf['truth'][:]
pred = hf['pred'][:]
hf.close()
DATA_DF = pd.read_csv('app_summary.csv')

#

def create_scatter(data_df=DATA_DF):
    fig = px.scatter(data_df, x='x',
                     y='y', title='UMAP', opacity=0.3,
                     color='IDR')
                     # , color_continuous_scale='viridis'
    # fig.add_scatter(x=np.arange(0, 4), y=np.arange(0, 4), mode='lines', hoverinfo='skip', name='')
    return fig


#######################*************************
fig = create_scatter()

@app.callback(
    dash.dependencies.Output('profile', 'figure'),
    [dash.dependencies.Input('predictions', 'hoverData')])
def update_profile(hoverData, data_df=DATA_DF, bin_size=256):
    print(hoverData)
    seq_n = hoverData['points'][0]['pointIndex']
    true_profile = np.repeat(truth[seq_n,:], bin_size)
    pred_profile = np.repeat(pred[seq_n,:], bin_size)
    pr = scipy_pr(true_profile, pred_profile)
    coord = data_df['coordinates'][seq_n]
    print(coord)
    fig = px.line(x=np.arange(len(true_profile)),
                  y = true_profile, title='{}; PR={}'.format(coord, str(np.round(pr, 3))))
    fig.add_scatter(x=np.arange(len(pred_profile)),
                    y = pred_profile,
                    name='Predicted', mode='lines', line=go.scatter.Line(color="red"))
    fig.update_layout()
    return fig

#######################*************************



app.layout = html.Div([
    # main scatter plot panel
    html.Div([dcc.Graph(id='predictions', figure=fig)],
              style={'width': '49%', 'display': 'inline-block'}),
    html.Div([
              dcc.Graph(id='profile')],
              style={'width': '49%', 'display': 'inline-block'})
    #
    # ])

    ])




if __name__ == '__main__':
    app.run_server(debug=True, port=8000)
