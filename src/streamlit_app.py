import sys
import os
from pathlib import Path
import json
import time
import threading

import streamlit as st
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ensure project root available for imports when Streamlit runs this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detect import Detector
import pyttsx3
from gtts import gTTS


st.set_page_config(page_title='Traffic Sign Recognition', layout='wide', initial_sidebar_state='expanded')

with open(PROJECT_ROOT / 'config.json','r') as f:
    cfg = json.load(f)

# --- Styling -------------------------------------------------
st.markdown(
    "<style>"
    "body{background:#0f1724;}"
    ".stApp {color: #e6eef8;}"
    ".big-title {font-size:32px; font-weight:700; color:#fff;}"
    ".small {color:#cdd6e0;}"
    "</style>", unsafe_allow_html=True)

st.markdown('<div class="big-title">Traffic Sign Recognition — Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="small">ViT + YOLO hybrid | Real-time detection | Voice alerts (EN/TA)</div>', unsafe_allow_html=True)

# layout
left, right = st.columns([2,1])

# Controls
with right:
    st.header('Controls')
    confidence_th = st.slider('Confidence threshold', 0.1, 0.99, float(cfg['detection']['confidence_threshold']), 0.01)
    use_yolo = st.checkbox('Enable YOLO detector', value=bool(cfg['detection'].get('yolo_model')))
    start_stream = st.button('Start Webcam')
    stop_stream = st.button('Stop Webcam')
    st.markdown('---')
    st.subheader('Recent detections')
    try:
        df_all = pd.read_csv(cfg['paths']['logs'])
        st.dataframe(df_all.tail(10))
    except Exception:
        st.write('No logs yet')

# placeholders
frame_placeholder = left.empty()
charts_placeholder = left.empty()

# voice engine
engine = pyttsx3.init()

def speak(text, lang='en'):
    if not text:
        return
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception:
        try:
            tts = gTTS(text=text, lang=lang)
            tmp = str(PROJECT_ROOT / 'temp_voice.mp3')
            tts.save(tmp)
            if os.name == 'nt':
                os.startfile(tmp)
            else:
                os.system(f'xdg-open "{tmp}"')
        except Exception:
            pass


detector = Detector(config_path=str(PROJECT_ROOT / 'config.json'))

running = False
thread_obj = None


def stream_thread():
    global running
    cap = cv2.VideoCapture(0)
    # discover classes from dataset/train
    train_dir = cfg['dataset']['train_dir']
    classes = []
    try:
        classes = [d.name for d in Path(train_dir).iterdir() if d.is_dir()]
        classes.sort()
    except Exception:
        # fallback if dataset not present
        classes = ['speed_limit_20','stop','yield','no_entry','turn_right']
    detector.load_vit(num_classes=len(classes))
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        outs = detector.detect_frame(frame, classes)
        for o in outs:
            x1,y1,x2,y2 = o['box']
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,200,0),2)
            cv2.putText(frame, f"{o['class']}:{o['confidence']:.2f}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)
            # voice alerts
            if o['confidence'] > confidence_th:
                speak(o.get('meaning_en',''), lang='en')
                speak(o.get('meaning_ta',''), lang='ta')

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, channels='RGB')
        # update small charts
        try:
            df = pd.read_csv(cfg['paths']['logs'])
            with charts_placeholder.container():
                st.subheader('Class distribution (last 100)')
                dist = df['class'].value_counts().head(10)
                st.bar_chart(dist)
                st.subheader('Confidence over time (last 50)')
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                st.line_chart(df.tail(50)['confidence'])
        except Exception:
            pass

    cap.release()


if start_stream:
    if thread_obj is None or not thread_obj.is_alive():
        running = True
        thread_obj = threading.Thread(target=stream_thread, daemon=True)
        thread_obj.start()

if stop_stream:
    running = False

