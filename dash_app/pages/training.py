from dash import html, dcc
import dash_bootstrap_components as dbc

layout = dbc.Container([
    html.H2('Model Training'),
    dbc.Row([
        dbc.Col([
            html.H5('Model Selection'),
            dcc.Dropdown(id='model-select', options=[{'label':'ViT','value':'vit'},{'label':'ResNet','value':'resnet'},{'label':'MobileNet','value':'mobilenet'}], value='vit'),
            html.H5('Hyperparameters', className='mt-3'),
            html.Label('Epochs'),
            dcc.Slider(id='epochs', min=1, max=100, step=1, value=10),
            html.Label('Learning Rate'),
            dcc.Slider(id='lr', min=1e-6, max=1e-1, step=1e-6, value=1e-4),
            html.Label('Batch Size'),
            dcc.Slider(id='batch', min=4, max=128, step=4, value=32),
            dbc.Button('Start Training', id='start-train', color='success', className='mt-3'),
            html.Div(id='training-status', className='mt-2')
        ], width=4),
        dbc.Col([
            html.H5('Live Progress'),
            dcc.Graph(id='train-curve'),
            dcc.Graph(id='val-curve'),
            dcc.Interval(id='training-interval', interval=2000, n_intervals=0)
        ], width=8)
    ])
], fluid=True)
