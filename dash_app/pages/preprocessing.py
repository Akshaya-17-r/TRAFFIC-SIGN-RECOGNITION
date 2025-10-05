from dash import html, dcc
import dash_bootstrap_components as dbc

layout = dbc.Container([
    html.H2('Dataset & Preprocessing'),
    dbc.Row([
        dbc.Col([
            html.H5('Load Dataset'),
            dcc.Upload(id='upload-data', children=html.Div(['Drag and Drop or ', html.A('Select Files')]), multiple=True),
            dbc.Button('Run Preprocessing', id='run-preprocess', color='primary', className='mt-3'),
            html.Div(id='preprocess-output', className='mt-2')
        ], width=4),
        dbc.Col([
            html.H5('Sample Before/After'),
            html.Img(id='sample-before', style={'maxWidth':'100%','border':'1px solid #ccc'}),
            html.Img(id='sample-after', style={'maxWidth':'100%','border':'1px solid #ccc','marginTop':'12px'})
        ], width=8)
    ])
], fluid=True)
