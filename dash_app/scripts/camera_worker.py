import threading
import time
import cv2
import os
import csv
from pathlib import Path
from .yolo_vit_detector import HybridDetector
from .voice import speak

_worker = None
_running = False

ASSETS_DIR = Path(__file__).resolve().parents[1] / 'assets'
LOGS_DIR = Path(__file__).resolve().parents[1] / 'logs'
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DETECTIONS_CSV = LOGS_DIR / 'detections.csv'
if not DETECTIONS_CSV.exists():
    with open(DETECTIONS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp','class','confidence','xmin','ymin','xmax','ymax'])


def _log_detection(row):
    with open(DETECTIONS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def start_camera(src=0, cfg_path=None, device='cpu', voice_lang='en'):
    global _worker, _running
    if _running:
        return False
    _running = True

    def _run():
        det = HybridDetector(cfg_path=cfg_path, device=device)
        cap = cv2.VideoCapture(src)
        vit_img = ASSETS_DIR / 'webcam_vit.jpg'
        yolo_img = ASSETS_DIR / 'webcam_yolo.jpg'
        try:
            while _running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                # run detector
                dets = det.detect(frame, conf_threshold=0.4)
                # draw boxes on a copy for vit and yolo panels (same currently)
                vit_frame = frame.copy()
                yolo_frame = frame.copy()
                for d in dets:
                    xmin,ymin,xmax,ymax = d['bbox']
                    label = d['label']
                    conf = d['conf']
                    cv2.rectangle(vit_frame, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
                    cv2.putText(vit_frame, f"{label} {conf:.2f}", (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.rectangle(yolo_frame, (xmin,ymin), (xmax,ymax), (0,0,255), 2)
                    cv2.putText(yolo_frame, f"{label} {conf:.2f}", (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    # log human-readable label
                    lbl = str(label)
                    _log_detection([time.time(), lbl, f"{conf:.4f}", xmin, ymin, xmax, ymax])
                    # speak short instruction for high-confidence detections
                    try:
                        if conf > 0.6:
                            # try to load meaning from sign_meanings.json
                            import json
                            from pathlib import Path
                            root = Path(__file__).resolve().parents[2]
                            meanings_path = root / 'dash_app' / 'sign_meanings.json'
                            text_to_speak = lbl
                            if meanings_path.exists():
                                try:
                                    meanings = json.loads(meanings_path.read_text())
                                    info = meanings.get(lbl, {})
                                    # prefer language-specific field if present
                                    text_to_speak = info.get('meaning', lbl)
                                except Exception:
                                    pass
                            # non-blocking speak with UI-selected language
                            speak(text_to_speak, lang=voice_lang or 'en')
                    except Exception:
                        pass

                cv2.imwrite(str(vit_img), vit_frame)
                cv2.imwrite(str(yolo_img), yolo_frame)
                time.sleep(0.03)
        finally:
            cap.release()

    _worker = threading.Thread(target=_run, daemon=True)
    _worker.start()
    return True


def stop_camera():
    global _running
    _running = False
    return True
