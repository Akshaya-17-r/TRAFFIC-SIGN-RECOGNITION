from dash import html
import dash_bootstrap_components as dbc

layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Img(src='/assets/traffic_logo.png', style={'width':'120px'}), width=2),
        dbc.Col(html.H1('Traffic Sign Recognition', style={'color':'#0b3d91'}), width=8),
    ], align='center', className='my-4'),
    dbc.Row([
        dbc.Col(html.Div([
            html.H3('Welcome'),
            html.P('A mini Driver Assistance prototype using ViT and Dash.'),
        ]), width=8),
        dbc.Col(html.Div([
            html.H5('Quick Links'),
            dbc.Button('Start Preprocessing', href='/preprocessing', color='primary', className='mb-2'),
            dbc.Button('Open Real-Time', href='/realtime', color='secondary')
        ]), width=4)
    ])
], fluid=True)
