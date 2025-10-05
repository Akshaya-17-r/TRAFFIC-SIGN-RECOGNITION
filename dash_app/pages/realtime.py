from dash import html, dcc
import dash_bootstrap_components as dbc
import json

with open('sign_meanings.json', 'r', encoding='utf-8') as f:
    SIGN_MEANINGS = json.load(f)


layout = dbc.Container([
    html.H2('Real-Time Detection'),
    dbc.Row([
        dbc.Col([
            html.H5('ViT (Classifier)'),
            html.Img(id='webcam-vit', src='/assets/placeholder.png', style={'width':'100%','border':'1px solid #ccc'})
        ], width=6),
        dbc.Col([
            html.H5('YOLO (Detector)'),
            html.Img(id='webcam-yolo', src='/assets/placeholder.png', style={'width':'100%','border':'1px solid #ccc'})
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col([
            html.Label('Voice language'),
            dcc.Dropdown(id='voice-lang', options=[{'label':'English','value':'en'},{'label':'Tamil','value':'ta'},{'label':'Hindi','value':'hi'}], value='en')
        ], width=4),
        dbc.Col([
            dbc.Button('Start', id='start-cam', color='primary'),
            dbc.Button('Stop', id='stop-cam', color='danger', className='ms-2'),
            dbc.Button('Download YOLO', id='download-yolo', color='secondary', className='ms-2'),
            html.Div(id='download-status', className='mt-2')
        ], width=4)
    ], className='my-3'),
    dbc.Row([
        dbc.Col(html.Div(id='detection-list'), width=12)
    ]),

    # Modal for explanation
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(id='explain-title')),
        dbc.ModalBody(id='explain-body'),
        dbc.ModalFooter(dbc.Button('Close', id='close-explain', className='ms-auto'))
    ], id='explain-modal', size='lg')

], fluid=True)

