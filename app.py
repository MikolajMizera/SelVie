"""
SelVie is a Dash application to display a set of molecular structures, enabling 
filtering by a specified value using a slider.

The app is driven by an SDF file with molecular structures and respective 
properties:
    
    mol_id: string
    An identifier for a molecule.

    `receptor_name`_predicted: float
    This property indicates a predicted value for a specified receptor.
    Example: use the property name `D1_predicted` - to indicate predicted value 
    for the D1 receptor.
    
    `receptor_name`_experiemntal: numeric or N/A (optional)
    This property indicates an experimentally measured value for a specified 
    receptor.
    Example: use property name `D1_experiemntal` - to indicate experimental 
    value for the D1 receptor.
    
    `receptor_name`_error: numeric or N/A (optional)
    This property indicates a prediction error for a specified receptor.
    Example: use the property name `D1_error` - to indicate prediction error 
    for the D1 receptor.
    
    vendor: string or N/A (optional)
    This property contains a vendor name.
    
    price: float or N/A (optional)
    This property contains the price for some fixed quantity of substance.
    
    db_url: string (optional)
    An URL address to an external database.
    
    db_name: string (optional)
    An URL address to an external database.    
"""

from os.path import join
import numpy as np
import pandas as pd
from rdkit import Chem

import matplotlib.pyplot as plt
import seaborn as sns

import dash
from dash.exceptions import PreventUpdate

from dash.dependencies import Input, Output, State
import dash_table
import dash_core_components as dcc
import dash_html_components as html

from utils import parse_sdf, draw_base64, preprocess_mols

from time import time

__author__ = 'Mikołaj Mizera'
__copyright__ = 'Copyright 2020, SMViewer'
__license__ = 'GNU General Public License v3.0'
__version__ = '0.0.1'
__maintainer__ = 'Mikołaj Mizera'
__email__ = 'mikolajmizera@gmail.com'
__status__ = 'PROTOTYPE'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

upload_div = html.Div([
        dcc.Upload(
                html.Div([
                    'Drag and drop SDF file or ',
                    html.A('Select SDF file')                        
                ]),
                id='upload_file',
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },    
        ),
    html.Div(id='output-data'),
    ])

data_table = dash_table.DataTable(id='table', 
                                  data=[],
                                  merge_duplicate_headers=True)

filters_div = html.Div([
                html.Table([
                        html.Tr([
                                html.Td(dcc.Dropdown(id='main_rec_drop')),
                                html.Td(dcc.Dropdown(id='secondary_rec_drop',
                                                     multi=True))
                                ]),
                        html.Tr([
                                html.Td(dcc.RangeSlider(id='sel_slider',
                                                        min=0,
                                                        max=1,
                                                        step=0.01,
                                                        value=[0, 1],
                                                        marks={0:{'label':'0'},
                                                               1:{'label':'1'}}),
                                        colSpan=2),
                                ])
                            ], style = {'width': '100%'}),                
                                
                ], style={'width': '600px'})

app.layout = html.Div([
        upload_div,
        dcc.Store(id='df-store', storage_type='memory'),
        filters_div,
        data_table])


@app.callback(Output('df-store', 'data'),
              [Input('upload_file', 'contents')],
              [State('upload_file', 'filename'),
               State('upload_file', 'last_modified')])
def process_upload(contents, name, date):
    if contents is not None:
        mols, msg, sess_id = parse_sdf(contents, name)
        df = preprocess_mols(mols, sess_id).astype(float)
        jsonfied = df.to_json(double_precision=3)
        return jsonfied
    else:
        return []

@app.callback([Output('table', 'data'), 
               Output('table', 'columns')],
              [Input('df-store', 'data')])
def update_table(data):
    
    try:
        df = pd.read_json(data)
    except:
        raise PreventUpdate
    
    cols = [{'name': n.split('_'), 'id': n} for n in df.columns]
    return df.round(3).to_dict('records'), cols

@app.callback([Output('main_rec_drop', 'options'),
               Output('main_rec_drop', 'value')],
              [Input('table', 'columns')])
def update_main_rec_drop(columns):
    
    columns = [c['id'] for c in columns]
    receptors = np.unique([c.split('_')[0] for c in columns])
    options = [{'label': r, 'value': r} for r in receptors]
    return options, receptors[0]

@app.callback([Output('secondary_rec_drop', 'options'),
               Output('secondary_rec_drop', 'value')],
              [Input('main_rec_drop', 'value')],
              [State('table', 'columns')])
def update_secondary_rec_drop(main_rec, columns):
        
    columns = [c['id'] for c in columns]
    receptors = np.unique([c.split('_')[0] for c in columns])
    options = [{'label': r, 'value': r} for r in receptors if r!=main_rec]
    return options, receptors[1:]

@app.callback([Output('sel_slider', 'min'),
               Output('sel_slider', 'max'),
               Output('sel_slider', 'value'),
               Output('sel_slider', 'marks')],
              [Input('secondary_rec_drop', 'value')],
              [State('table', 'data'),
               State('main_rec_drop', 'value')])
def update_sel_slider(sec_rec, selected_df, main_rec):
        
    if (main_rec is None) or (sec_rec is None) or (selected_df is None):
        raise PreventUpdate
    
    df = pd.DataFrame(selected_df)
    preds_df = df[[c for c in df.columns if 'prediction' in c]]
    
    isin_cols = lambda x: [r in x for r in sec_rec]
    sec_rec_cols = [c for c in preds_df.columns if np.any(isin_cols(c))]
    sec_rec_data = preds_df[sec_rec_cols]
    
    deltas = (-(sec_rec_data.T-preds_df['%s_prediction'%main_rec]).T).min(1)
    max_val = deltas.max()
    
    min_mark, max_mark =  [0, max_val] 
    marks = {n:{'label':'%.1f'%n} for n in np.arange(min_mark, max_mark, 0.2)} 
    
    return 0, max_val, [0, max_val], marks               

if __name__ == '__main__':
    app.run_server(debug=False)