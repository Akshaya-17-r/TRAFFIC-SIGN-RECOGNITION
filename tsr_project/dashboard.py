"""
Streamlit dashboard for the TSR project.

Features:
- Live webcam stream with detection overlays (via Detector)
- Display last 10 logs and charts
- Adjustable confidence threshold and start/stop controls
"""
import streamlit as st
import pandas as pd
import cv2
import threading
import time
import json

from tsr_project.detect import Detector


st.set_page_config(page_title="Traffic Sign Recognition", layout="wide")


@st.cache_resource
def get_detector():
    det = Detector()
    # Try to infer classes from dataset train folder
    try:
        with open("config.json") as f:
            cfg = json.load(f)
        train_dir = cfg["dataset"]["train_dir"]
        classes = sorted([d for d in __import__("os").listdir(train_dir) if __import__("os").path.isdir(__import__("os").path.join(train_dir, d))])
        det.load_vit(classes)
    except Exception:
        pass
    return det


def main():
    st.title("Traffic Sign Recognition Dashboard")
    st.markdown("A mini Driver Assistance System prototype using YOLO + ViT")

    detector = get_detector()

    col1, col2 = st.columns([2,1])

    with col2:
        st.header("Controls")
        conf = st.slider("Confidence threshold", 0.0, 1.0, float(detector.conf_threshold), 0.01)
        if st.button("Start Webcam"):
            st.session_state["running"] = True
        if st.button("Stop Webcam"):
            st.session_state["running"] = False

        st.write("Last detections")
        try:
            df = pd.read_csv(detector.log_path)
            st.dataframe(df.tail(10))
        except Exception:
            st.info("No logs yet.")

    with col1:
        st.header("Live Stream")
        img_placeholder = st.empty()

    if "running" not in st.session_state:
        st.session_state["running"] = False

    def stream():
        cap = cv2.VideoCapture(0)
        while st.session_state["running"] and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            detector.conf_threshold = conf
            detections = detector.detect_frame(frame)
            # Draw
            for d in detections:
                xmin, ymin, xmax, ymax = d["bbox"]
                label = d["label"]
                confv = d["conf"]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                cv2.putText(frame, f"{label} {confv:.2f}", (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_placeholder.image(frame_rgb, channels="RGB")
            time.sleep(0.03)

        cap.release()

    if st.session_state["running"]:
        t = threading.Thread(target=stream, daemon=True)
        t.start()


if __name__ == "__main__":
    main()
