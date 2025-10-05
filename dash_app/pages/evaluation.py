from dash import html, dcc
import dash_bootstrap_components as dbc

layout = dbc.Container([
    html.H2('Evaluation'),
    dbc.Row([
        dbc.Col([
            dcc.Upload(id='upload-test', children=html.Div(['Upload Test Set']), multiple=False),
            dbc.Button('Run Evaluation', id='run-eval', color='primary', className='mt-2'),
            html.Div(id='eval-output', className='mt-2')
        ], width=4),
        dbc.Col([
            html.H5('Confusion Matrix'),
            html.Img(id='confusion-img', src='/assets/placeholder.png', style={'width':'100%','border':'1px solid #ccc'}),
            html.H5('Classification Report'),
            html.Pre(id='eval-report', style={'whiteSpace':'pre-wrap','background':'#f7f7f7','padding':'8px','borderRadius':'4px'}),
            html.H5('Misclassified Samples'),
            html.Div(id='misclassified-gallery', style={'display':'flex','flexWrap':'wrap','gap':'8px','marginTop':'8px'}),
            html.H5('Most Confused Pairs'),
            dcc.Graph(id='most-confused')
        ], width=8)
    ])
], fluid=True)
