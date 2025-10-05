"""
Detection pipeline combining YOLO (for bounding boxes) and ViT (for per-crop re-classification).

Provides real-time webcam detection and logging to CSV.
"""
import os
import time
import csv
import json
import threading
from datetime import datetime

import cv2
import torch
from ultralytics import YOLO
import numpy as np

from tsr_project.model import create_vit_model
from tsr_project.voice import VoiceAlert


class Detector:
    def __init__(self, config_path="config.json"):
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.device = self.cfg["detection"].get("device", "cpu")
        self.conf_threshold = self.cfg["detection"].get("confidence_threshold", 0.5)
        self.yolo_path = self.cfg["detection"].get("yolo_model", "yolov8n.pt")

        # Load YOLO model (ultralytics)
        self.yolo = YOLO(self.yolo_path)

        # Placeholder: ViT model will be loaded lazily after classes are known
        self.vit = None
        self.classes = None

        self.log_path = self.cfg["paths"].get("logs", "logs/detections.csv")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "class", "confidence", "xmin", "ymin", "xmax", "ymax"])

        self.voice = VoiceAlert()

    def load_vit(self, classes, model_name="vit_base_patch16_224"):
        self.classes = classes
        self.vit = create_vit_model(num_classes=len(classes), pretrained=True, model_name=model_name)
        self.vit.to(self.device)

    def detect_frame(self, frame):
        """Run YOLO detection then refine with ViT on each crop."""
        results = self.yolo(frame)[0]
        detections = []
        for box, score, cls in zip(results.boxes.xyxy.tolist(), results.boxes.conf.tolist(), results.boxes.cls.tolist()):
            if score < self.conf_threshold:
                continue
            xmin, ymin, xmax, ymax = map(int, box)
            crop = frame[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue
            # Preprocess crop for ViT: resize to 224 and normalize
            crop_resized = cv2.resize(crop, (224, 224))
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            input_tensor = torch.from_numpy(crop_rgb).float().permute(2,0,1)/255.0
            # normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            input_tensor = (input_tensor - mean)/std
            input_tensor = input_tensor.unsqueeze(0).to(self.device)

            if self.vit is not None:
                with torch.no_grad():
                    outputs = self.vit(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, dim=1)
                    conf = float(conf.cpu().item())
                    pred = int(pred.cpu().item())
                    label = self.classes[pred]
            else:
                conf = score
                label = str(int(cls))

            detections.append({"label": label, "conf": conf, "bbox": (xmin, ymin, xmax, ymax)})

            # Logging
            self.log_detection(label, conf, xmin, ymin, xmax, ymax)

            # Voice alert
            text_en = label
            instruction = ""
            try:
                with open("sign_meanings.json") as f:
                    meanings = json.load(f)
                if label in meanings:
                    text_en = meanings[label].get("en", label)
                    instruction = meanings[label].get("instruction", "")
            except Exception:
                pass

            # If low confidence, use fail-safe message
            if conf < self.conf_threshold:
                self.voice.speak(f"Possible {text_en}. Low confidence.")
            else:
                self.voice.speak(text_en)
                if instruction:
                    self.voice.speak(instruction)

        return detections

    def log_detection(self, label, conf, xmin, ymin, xmax, ymax):
        ts = datetime.utcnow().isoformat()
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, label, f"{conf:.4f}", xmin, ymin, xmax, ymax])


def run_webcam(detector: Detector, src=0):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = detector.detect_frame(frame)
            # Draw detections
            for d in detections:
                xmin, ymin, xmax, ymax = d["bbox"]
                label = d["label"]
                conf = d["conf"]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.imshow("TSR Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    det = Detector()
    run_webcam(det)
