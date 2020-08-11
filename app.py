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

from ast import literal_eval
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

from utils import parse_sdf, draw_base64, preprocess_mols, load_props

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

upload_td = html.Td([
        dcc.Upload(
                html.Div([
                    'Drag and drop SDF file or ',
                    html.A('select SDF file')                        
                ]),
                id='upload_file',
                style={
                    'width': '300px',
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

filters_td = html.Td([
                html.Tr([
                        html.Td(dcc.Dropdown(id='main_rec_drop'),
                                style={'width': '100px'}),
                        html.Td(dcc.Dropdown(id='secondary_rec_drop',
                                             multi=True),
                                style={'width': '400px'})
                        ]),
                html.Tr([
                        html.Td(dcc.RangeSlider(id='sel_slider',
                                                min=0,
                                                max=1,
                                                step=0.01,
                                                value=[0, 1],
                                                marks={0:{'label':'0'},
                                                       1:{'label':'1'}}),
                                colSpan=2,
                                style={'border': 'none'}),
                        ])
            ])

sim_td = html.Td(dcc.RangeSlider(id='sim_slider',
                                  min=0,
                                  max=1,
                                  step=0.01,
                                  value=[0, 1],
                                  marks={0:{'label':'0'},
                                         1:{'label':'1'}}),
                style={'width': '400px'})

app.layout = html.Div([
        dcc.Store(id='df-store', storage_type='memory'),
        html.Table([html.Tr([html.Th('Upload'), 
                            html.Th('Filter by predicted selectivity'), 
                            html.Th('Filter by similarity')]),
                    html.Tr([upload_td, filters_td, sim_td])],
                   style={'margin-bottom': '10px'}),        
        data_table])


@app.callback(Output('df-store', 'data'),
              [Input('upload_file', 'contents')],
              [State('upload_file', 'filename'),
               State('upload_file', 'last_modified')])
def process_upload(contents, name, date):
    if contents is not None:
        mols, msg, sess_id = parse_sdf(contents, name)
        sess_dir = preprocess_mols(mols, sess_id)
        df = load_props(sess_dir)
        del df['NN']
        jsonfied = df.to_json(double_precision=3)
        return jsonfied
    else:
        return []

@app.callback([Output('main_rec_drop', 'options'),
               Output('main_rec_drop', 'value')],
              [Input('df-store', 'data')])
def update_main_rec_drop(data):
    
    try:
        df = pd.read_json(data)
    except:
        raise PreventUpdate
       
    receptors = np.unique([c.split('_')[0] for c in df.columns 
                           if 'prediction' in c])
    options = [{'label': r, 'value': r} for r in receptors]
    return options, receptors[0]

@app.callback([Output('secondary_rec_drop', 'options'),
               Output('secondary_rec_drop', 'value')],
              [Input('main_rec_drop', 'value')],
              [State('df-store', 'data')])
def update_secondary_rec_drop(main_rec, data):
    
    try:
        df = pd.read_json(data)
    except:
        raise PreventUpdate
        
    receptors = np.unique([c.split('_')[0] for c in df.columns
                           if 'prediction' in c])
    options = [{'label': r, 'value': r} for r in receptors if r!=main_rec]
    return options, [r['value'] for r in options]

@app.callback([Output('sel_slider', 'min'),
               Output('sel_slider', 'max'),
               Output('sel_slider', 'value'),
               Output('sel_slider', 'marks'),
               Output('sim_slider', 'min'),
               Output('sim_slider', 'max'),
               Output('sim_slider', 'value'),
               Output('sim_slider', 'marks')],
              [Input('secondary_rec_drop', 'value')],
              [State('main_rec_drop', 'value'),
               State('df-store', 'data')])
def update_sliders(sec_rec, main_rec, data):
    
    try:
        selected_df = pd.read_json(data)
    except:
        raise PreventUpdate
        
    fired_by = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    
    ## Debug info
    print(fired_by)
    print(sec_rec, main_rec)
    
    if (main_rec is None) or (selected_df is None) or (sec_rec is None):
        raise PreventUpdate
    
    df = pd.DataFrame(selected_df)
    
    if fired_by == 'main_rec_drop':
        columns = [c for c in df.columns]
        receptors = np.unique([c.split('_')[0] for c in columns])
        sec_rec = [r for r in receptors if r!=main_rec]
    
    preds_df = df[[c for c in df.columns if 'prediction' in c]]
    
    isin_cols = lambda x: [r in x for r in sec_rec]
    sec_rec_cols = [c for c in preds_df.columns if np.any(isin_cols(c))]
    sec_rec_data = preds_df[sec_rec_cols]
    
    deltas = (-(sec_rec_data.T-preds_df['%s_prediction'%main_rec]).T).min(1)
    max_val = deltas.max()
    
    min_mark, max_mark =  [0, max_val] 
    marks = {n:{'label':'%.1f'%n} for n in np.arange(min_mark, max_mark, 0.5)} 
    
    sel_updates = (0, max_val, [0, max_val], marks)
    
    similarities = df['Similarity_Tanimoto']
    
    max_val = similarities.max()
    
    min_mark, max_mark =  [0, max_val] 
    marks = {n:{'label':'%.1f'%n} for n in np.arange(min_mark, max_mark, 0.2)} 
    
    sim_updates = (0, max_val, [0, max_val], marks)
    
    return  sel_updates+sim_updates


@app.callback([Output('table', 'data'), 
               Output('table', 'columns')],
              [Input('sel_slider', 'value'),
               Input('sim_slider', 'value')],
               [State('main_rec_drop', 'value'),
                State('secondary_rec_drop', 'value'),
                State('df-store', 'data')])
def update_table(sel_range, sim_range, main_rec, sec_rec, data):
    
    try:
        df = pd.read_json(data)
    except:
        raise PreventUpdate
    
    # Apply selectivity filter
    min_sel, max_sel = sel_range
    
    preds_df = df[[c for c in df.columns if 'prediction' in c]]
    
    isin_cols = lambda x: [r in x for r in sec_rec]
    sec_rec_cols = [c for c in preds_df.columns if np.any(isin_cols(c))]
    sec_rec_data = preds_df[sec_rec_cols]
    
    deltas = (-(sec_rec_data.T-preds_df['%s_prediction'%main_rec]).T).min(1)
    deltas_mask = deltas >= min_sel
    deltas_mask &= deltas <= max_sel
    
    # Apply similarity filter
    min_sim, max_sim = sim_range
    sim_mask = df['Similarity_Tanimoto'] >= min_sim
    sim_mask &= df['Similarity_Tanimoto'] <= max_sim
    
    cols = [{'name': n.split('_'), 'id': n} for n in df.columns]
    selected_df = df[sim_mask&deltas_mask].round(1).to_dict('records')

    return selected_df, cols

if __name__ == '__main__':
    app.run_server(debug=False)