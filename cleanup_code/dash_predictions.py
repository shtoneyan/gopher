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
import os, sys

# y axis fixed somewhere in the Average
# remove mse
# dropdown menu for cell lines
# plot extra cell lines

CELL_LINE = int(sys.argv[1])

run_path = sys.argv[2] #'/home/shush/profile/QuantPred/wandb/run-20210623_073933-vwq5gdk5/files/'
h5_path = os.path.join(run_path, str(CELL_LINE) + '_dash.h5')


server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)


hf = h5py.File(h5_path, 'r')
truth = hf['y_true'][:]
pred = hf['y_pred'][:]
bin_size = hf['bin_size'][()]
print(bin_size)
hf.close()
# DATA_DF = pd.read_csv('app_summary.csv')

DATA_DF = pd.DataFrame(np.array([(truth.mean(axis=1)),
 (pred.mean(axis=1))]).T, columns=['Experimental', 'Predicted'])

def create_scatter():
    fig = px.scatter(DATA_DF, x='Experimental',
                     y='Predicted', title='Average coverage', opacity=0.3)
                     # , color_continuous_scale='viridis'
    # fig.add_scatter(x=np.arange(0, 4), y=np.arange(0, 4), mode='lines', hoverinfo='skip', name='')
    return fig


#######################*************************
fig = create_scatter()

# @app.callback(
#     dash.dependencies.Output('profile', 'figure'),
#     [dash.dependencies.Input('predictions', 'hoverData')])
# def update_profile(hoverData):
#     seq_n = hoverData['points'][0]['pointIndex']
#     true_profile = np.repeat(truth[seq_n,:], bin_size)
#     pred_profile = np.repeat(pred[seq_n,:], bin_size)
#
#     fig = px.line(x=np.arange(len(true_profile)),
#                   y = true_profile)
#     fig.add_scatter(x=np.arange(len(pred_profile)),
#                     y = pred_profile,
#                     name='Predicted', mode='lines', line=go.scatter.Line(color="red"))
#     fig.update_layout()
#     return fig


@app.callback(
    dash.dependencies.Output('profile_truth', 'figure'),
    [dash.dependencies.Input('predictions', 'hoverData')])
def update_profile(hoverData):
    seq_n = hoverData['points'][0]['pointIndex']
    true_profile = np.repeat(truth[seq_n,:], bin_size)

    fig = px.line(x=np.arange(len(true_profile)),
                  y = true_profile)
    fig.update_layout()
    return fig


@app.callback(
    dash.dependencies.Output('profile_pred', 'figure'),
    [dash.dependencies.Input('predictions', 'hoverData')])
def update_profile(hoverData):
    seq_n = hoverData['points'][0]['pointIndex']
    pred_profile = np.repeat(pred[seq_n,:], bin_size)

    fig = px.line(x=np.arange(len(pred_profile)),
                  y = pred_profile)

    fig.update_layout()
    return fig


#######################*************************



app.layout = html.Div([
    # main scatter plot panel
    html.Div([dcc.Graph(id='predictions', figure=fig)],
              style={'width': '49%', 'display': 'inline-block'}),
    html.Div([
              dcc.Graph(id='profile_truth')],
              style={'width': '49%', 'display': 'inline-block'}),
    html.Div([
              dcc.Graph(id='profile_pred')],
              style={'width': '49%', 'display': 'inline-block'})
    ])




if __name__ == '__main__':
    app.run_server(debug=True, port=8000)
