import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pages.landing as landing
import pages.preprocessing as preprocessing
import pages.training as training
import pages.evaluation as evaluation
import pages.realtime as realtime
import pages.knowledge as knowledge

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    dbc.NavbarSimple(
        brand='Traffic Sign Recognition Dashboard',
        color='dark',
        dark=True,
        children=[
            dbc.NavItem(dbc.NavLink('Home', href='/')),
            dbc.NavItem(dbc.NavLink('Preprocessing', href='/preprocessing')),
            dbc.NavItem(dbc.NavLink('Training', href='/training')),
            dbc.NavItem(dbc.NavLink('Evaluation', href='/evaluation')),
            dbc.NavItem(dbc.NavLink('Real-Time', href='/realtime')),
            dbc.NavItem(dbc.NavLink('Knowledge', href='/knowledge')),
        ]
    ),
    html.Div(id='page-content')
], fluid=True)


@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/preprocessing':
        return preprocessing.layout
    if pathname == '/training':
        return training.layout
    if pathname == '/evaluation':
        return evaluation.layout
    if pathname == '/realtime':
        return realtime.layout
    if pathname == '/knowledge':
        return knowledge.layout
    return landing.layout


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
import json
from pathlib import Path
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

ROOT = Path(__file__).resolve().parent

external_stylesheets = [dbc.themes.CYBORG]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

with open(ROOT / 'sign_meanings.json','r') as f:
    SIGN_MEANINGS = json.load(f)

nav = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink('Dataset', href='/dataset')),
        dbc.NavItem(dbc.NavLink('Training', href='/training')),
        dbc.NavItem(dbc.NavLink('Evaluation', href='/evaluation')),
        dbc.NavItem(dbc.NavLink('Real-time', href='/realtime')),
        dbc.NavItem(dbc.NavLink('Knowledge Base', href='/kb')),
    ],
    brand='Traffic Sign Recognition',
    color='dark',
    dark=True,
)

layout = html.Div([
    dcc.Location(id='url'),
    nav,
    html.Div(id='page-content', style={'padding':'24px'})
])

app.layout = layout


@app.callback(Output('page-content','children'), Input('url','pathname'))
def display_page(pathname):
    if pathname == '/dataset':
        return dataset_layout()
    if pathname == '/training':
        return training_layout()
    if pathname == '/evaluation':
        return evaluation_layout()
    if pathname == '/realtime':
        return realtime_layout()
    if pathname == '/kb':
        return kb_layout()
    return landing_layout()


def landing_layout():
    return html.Div([
        html.Div(className='card', children=[
            html.H1('Traffic Sign Recognition Dashboard', className='title'),
            html.P('ViT + YOLO hybrid prototype. Use the navigation to explore dataset, train models, evaluate, and run real-time detection.', className='muted')
        ])
    ])


def dataset_layout():
    return html.Div([
        html.Div(className='card', children=[
            html.H2('Dataset & Preprocessing'),
            dcc.Upload(id='upload-dataset', children=html.Div(['Drag and drop dataset or click to upload'])),
            html.Button('Run Preprocessing', id='run-preprocess', n_clicks=0),
            html.Div(id='preprocess-output')
        ])
    ])


def training_layout():
    return html.Div([
        html.Div(className='card', children=[
            html.H2('Model Training'),
            html.Div([html.Label('Model'), dcc.Dropdown(id='model-select', options=[{'label':'ViT','value':'vit'},{'label':'ResNet','value':'resnet'},{'label':'MobileNet','value':'mobilenet'}], value='vit')]),
            html.Div([html.Label('Epochs'), dcc.Slider(id='epochs-slider', min=1, max=50, step=1, value=10)]),
            html.Div([html.Label('Learning Rate'), dcc.Input(id='lr-input', type='number', value=1e-4)]),
            html.Button('Start Training', id='start-train', n_clicks=0),
            dcc.Graph(id='train-curve')
        ])
    ])


def evaluation_layout():
    return html.Div([
        html.Div(className='card', children=[
            html.H2('Evaluation'),
            html.Button('Run Evaluation', id='run-eval', n_clicks=0),
            html.Div(id='eval-output')
        ])
    ])


def realtime_layout():
    return html.Div([
        html.Div(className='card', children=[
            html.H2('Real-time Detection'),
            html.Button('Start Camera', id='start-cam', n_clicks=0),
            html.Button('Stop Camera', id='stop-cam', n_clicks=0),
            html.Div(id='camera-output')
        ])
    ])


def kb_layout():
    cards = []
    for k,v in SIGN_MEANINGS.items():
        cards.append(html.Div(className='card', children=[html.H4(k), html.P(v['en']), html.P(v['instruction'])]))
    return html.Div(cards)


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
