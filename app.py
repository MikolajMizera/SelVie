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

data_table = dash_table.DataTable(
                                id='table',
                                columns=[],
                                merge_duplicate_headers=True,
                                data={},
                                row_selectable='single',
                                selected_row_ids=[0],
                                fixed_rows={'headers': True, 
                                            'data': 0},
                                style_header={'backgroundColor': 'white',
                                              'fontWeight': 'bold'},
                                style_cell={'padding': '5px', 
                                            'minWidth': '40px', 
                                            'width': '40px', 
                                            'maxWidth': '40px'},
                                filter_action='native',
                                sort_action='native',
                                style_table={'maxHeight': '400px'})

app.layout = upload_div


@app.callback(Output('output-data', 'children'),
              [Input('upload_file', 'contents')],
              [State('upload_file', 'filename'),
               State('upload_file', 'last_modified')])
def populate_table(contents, name, date):
    if contents is not None:
        mols, msg, sess_id = parse_sdf(contents, name)
        df = preprocess_mols(mols, sess_id)
        return msg

if __name__ == '__main__':
    app.run_server(debug=False)