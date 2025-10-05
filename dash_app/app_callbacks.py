from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
from app import app
import dash
import json
from pathlib import Path
import shutil
import dash_bootstrap_components as dbc

ROOT = Path(__file__).resolve().parents[1]


@app.callback(
    Output('sample-before', 'src'),
    Output('sample-after', 'src'),
    Output('preprocess-output', 'children'),
    Input('run-preprocess', 'n_clicks')
)
def run_preprocess(n):
    if not n:
        raise PreventUpdate
    # Find a sample image from the train dataset
    try:
        cfg = json.loads((ROOT / 'config.json').read_text())
        train_dir = ROOT / cfg['dataset']['train_dir']
        # pick first class and first image
        classes = [d for d in train_dir.iterdir() if d.is_dir()]
        if not classes:
            return dash.no_update, dash.no_update, 'No classes found in dataset/train'
        first_class = classes[0]
        images = [p for p in first_class.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
        if not images:
            return dash.no_update, dash.no_update, 'No images found in first class'
        sample = images[0]

        # Copy original to assets/sample_before.jpg
        assets_dir = ROOT / 'dash_app' / 'assets'
        assets_dir.mkdir(parents=True, exist_ok=True)
        before_path = assets_dir / 'sample_before.jpg'
        after_path = assets_dir / 'sample_after.jpg'
        shutil.copy(sample, before_path)

        # Run preprocessing (use script in dash_app/scripts)
        try:
            from scripts import preprocessing as prep
            # preprocess_image will save the processed image
            prep.preprocess_image(str(sample), str(after_path))
        except Exception:
            # if preprocessing module not available, just reuse original as after
            shutil.copy(sample, after_path)

        # Return paths relative to assets for Dash static serving
        return '/assets/sample_before.jpg', '/assets/sample_after.jpg', 'Preprocessing completed (sample)'
    except Exception as e:
        return dash.no_update, dash.no_update, f'Error: {e}'


@app.callback(Output('train-curve','figure'), Input('start-train','n_clicks'), State('model-select','value'), State('epochs','value'))
def start_training(n, model, epochs):
    if not n:
        raise PreventUpdate
    # Here you would call models.train to run training in a background thread/process and stream updates
    # For demo, return a placeholder empty figure
    import plotly.express as px
    import pandas as pd
    df = pd.DataFrame({'epoch':[1,2,3],'loss':[1.2,0.8,0.5],'acc':[0.5,0.7,0.82]})
    fig = px.line(df, x='epoch', y=['loss','acc'], title='Training progress (demo)')
    return fig


@app.callback(Output('eval-output','children'), Input('run-eval','n_clicks'))
def run_eval(n):
    if not n:
        raise PreventUpdate
    try:
        from scripts.evaluate import evaluate_model, most_confused_pairs
        cfg = json.loads((Path(__file__).resolve().parents[1] / 'config.json').read_text())
        test_dir = cfg['dataset']['test_dir']
        assets_dir = Path(__file__).resolve().parents[1] / 'dash_app' / 'assets'
        assets_dir.mkdir(parents=True, exist_ok=True)
        cm, report = evaluate_model(None, test_dir, device='cpu', out_dir=str(assets_dir))

        # copy confusion matrix to assets path (already saved there by evaluate_model)
        conf_img = assets_dir / 'confusion_matrix.png'

        # build misclassified gallery
        mis_dir = assets_dir / 'misclassified'
        gallery = []
        if mis_dir.exists():
            for p in sorted(mis_dir.iterdir())[:20]:
                rel = '/assets/misclassified/' + p.name
                # ensure a copy exists in assets/misclassified
                gallery.append(rel)

        # most confused
        classes = []
        try:
            # infer classes from train folder
            train_dir = cfg['dataset']['train_dir']
            classes = sorted([d for d in Path(train_dir).iterdir() if d.is_dir()])
            classes = [c.name for c in classes]
        except Exception:
            classes = None
        most = most_confused_pairs(cm, classes if classes else [])

        import pprint
        rstr = pprint.pformat(report)
        # Return a tuple; downstream callbacks will update images and report areas if needed
        return 'Evaluation completed', str(conf_img), rstr, gallery, most
    except Exception as e:
        return f'Error running evaluation: {e}'


@app.callback(
    Output('confusion-img','src'),
    Output('eval-report','children'),
    Output('misclassified-gallery','children'),
    Output('most-confused','figure'),
    Input('run-eval','n_clicks')
)
def update_evaluation_ui(n):
    if not n:
        raise PreventUpdate
    try:
        res = run_eval(n)
        if isinstance(res, str):
            raise RuntimeError(res)
        status, conf_img_str, report_str, gallery, most = res
        # Build misclassified gallery elements
        gallery_elems = []
        for rel in gallery:
            gallery_elems.append(html.Img(src=rel, style={'width':'120px','height':'120px','objectFit':'cover','border':'1px solid #ccc'}))

        # build most confused figure
        import plotly.express as px
        import pandas as pd
        if most:
            df = pd.DataFrame(most, columns=['true','pred','count'])
            fig = px.bar(df, x='count', y=df.index.astype(str), orientation='h', labels={'y':'pair','x':'count'}, hover_data=['true','pred'])
        else:
            fig = go.Figure()

        return conf_img_str, report_str, gallery_elems, fig
    except Exception as e:
        raise PreventUpdate


@app.callback(Output('camera-output','children'), Input('start-cam','n_clicks'))
def start_cam(n):
    if not n:
        raise PreventUpdate
    return 'Camera started (demo)'


# Background training runner
from app_callbacks_worker import start_training_in_background, PROGRESS_PATH
import plotly.graph_objects as go
import pandas as pd
from scripts.voice import speak
import csv
from pathlib import Path
from scripts.camera_worker import start_camera, stop_camera
import threading
from scripts.download_yolo import download_yolo


DETECTIONS_CSV = Path(__file__).resolve().parents[1] / 'logs' / 'detections.csv'


@app.callback(Output('training-status','children'), Input('start-train','n_clicks'))
def start_training_cb(n):
    if not n:
        raise PreventUpdate
    # Start training in background
    start_training_in_background()
    return 'Training started in background'


@app.callback(Output('camera-output','children'), Input('start-cam','n_clicks'), State('voice-lang','value'))
def start_camera_cb(n, voice_lang):
    if not n:
        raise PreventUpdate
    # pass UI voice language into camera worker
    start_camera(cfg_path=str(Path(__file__).resolve().parents[1] / 'config.json'), voice_lang=voice_lang)
    return 'Camera started'


@app.callback(Output('camera-output','children'), Input('stop-cam','n_clicks'))
def stop_camera_cb(n):
    if not n:
        raise PreventUpdate
    stop_camera()
    return 'Camera stopped'


@app.callback(Output('webcam-vit','src'), Input('training-interval','n_intervals'))
def update_vit_image(n):
    p = Path(__file__).resolve().parents[1] / 'assets' / 'webcam_vit.jpg'
    if p.exists():
        return '/assets/webcam_vit.jpg?ts=' + str(int(p.stat().st_mtime))
    return dash.no_update


@app.callback(Output('webcam-yolo','src'), Input('training-interval','n_intervals'))
def update_yolo_image(n):
    p = Path(__file__).resolve().parents[1] / 'assets' / 'webcam_yolo.jpg'
    if p.exists():
        return '/assets/webcam_yolo.jpg?ts=' + str(int(p.stat().st_mtime))
    return dash.no_update


@app.callback(Output('download-status','children'), Input('download-yolo','n_clicks'))
def download_yolo_cb(n):
    if not n:
        raise PreventUpdate
    status_id = 'download-status'

    def _dl():
        try:
            path = download_yolo()
            # write small status file
            Path(__file__).resolve().parents[1].joinpath('dash_app','assets','yolo_downloaded.txt').write_text(path)
        except Exception as e:
            Path(__file__).resolve().parents[1].joinpath('dash_app','assets','yolo_downloaded.txt').write_text('error:'+str(e))

    threading.Thread(target=_dl, daemon=True).start()
    return 'Downloading YOLO in background...'


@app.callback(Output('train-curve','figure'), Input('training-interval','n_intervals'))
def poll_training(n):
    # read progress JSON if exists
    try:
        if not PROGRESS_PATH.exists():
            raise PreventUpdate
        data = json.loads(PROGRESS_PATH.read_text())
        epochs = data.get('epochs', [])
        if not epochs:
            raise PreventUpdate
        df = pd.DataFrame(epochs)
        fig = go.Figure()
        if 'train_loss' in df:
            fig.add_trace(go.Scatter(x=df['epoch'], y=df['train_loss'], name='train_loss'))
        if 'val_loss' in df:
            fig.add_trace(go.Scatter(x=df['epoch'], y=df['val_loss'], name='val_loss'))
        if 'train_acc' in df:
            fig.add_trace(go.Scatter(x=df['epoch'], y=df['train_acc'], name='train_acc', yaxis='y2'))
        if 'val_acc' in df:
            fig.add_trace(go.Scatter(x=df['epoch'], y=df['val_acc'], name='val_acc', yaxis='y2'))
        fig.update_layout(yaxis=dict(title='Loss'), yaxis2=dict(title='Accuracy', overlaying='y', side='right'))
        return fig
    except PreventUpdate:
        raise
    except Exception:
        # return empty figure
        return go.Figure()


@app.callback(Output('detection-list','children'), Input('training-interval','n_intervals'))
def poll_detections(n):
    # Read last 10 detections
    try:
        if not DETECTIONS_CSV.exists():
            return 'No detections yet.'
        rows = []
        with open(DETECTIONS_CSV, 'r', newline='') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for r in reader:
                rows.append(r)
        rows = rows[-10:][::-1]
        items = []
        for i, r in enumerate(rows):
            ts, cls, conf, *rest = r
            btn = dbc.Button(f"{cls} ({float(conf):.2f}) - {ts}", id={'type':'det-btn','index':i}, color='light', className='m-1')
            items.append(btn)
        return items
    except Exception:
        return 'Error reading detections.'


@app.callback(
    Output('explain-modal','is_open'),
    Output('explain-title','children'),
    Output('explain-body','children'),
    Input({'type':'det-btn','index':dash.ALL}, 'n_clicks'),
    State('voice-lang','value'),
    State('explain-modal','is_open')
)
def show_explanation(n_clicks_list, lang, is_open):
    # If any button clicked, find which
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    prop = ctx.triggered[0]['prop_id']
    try:
        # prop_id looks like {"type":"det-btn","index":0}.n_clicks
        btn_id = json.loads(prop.split('.')[0])
        idx = btn_id.get('index', 0)
    except Exception:
        raise PreventUpdate

    # Read last detections and pick index
    try:
        with open(DETECTIONS_CSV, 'r', newline='') as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows = list(reader)
        if not rows:
            raise PreventUpdate
        rows = rows[-10:][::-1]
        sel = rows[idx]
        ts, cls, conf, *rest = sel
        # Lookup meaning
        try:
            meanings = json.loads((Path(__file__).resolve().parents[1] / 'sign_meanings.json').read_text())
            info = meanings.get(cls, {})
            title = f"{cls} ({float(conf):.2f})"
            body = []
            body.append(html.P(info.get('meaning','Unknown')))
            body.append(html.P(html.Strong('Rule: ')+ info.get('rule','Not available')))
            body.append(html.P(html.Em(info.get('fact',''))))
        except Exception:
            title = cls
            body = [html.P('No info available')]

        # Speak the meaning in requested language
        text_to_speak = info.get('meaning', cls) + '. ' + info.get('instruction','')
        try:
            speak(text_to_speak, lang=lang)
        except Exception:
            pass

        return True, title, body
    except Exception:
        raise PreventUpdate
