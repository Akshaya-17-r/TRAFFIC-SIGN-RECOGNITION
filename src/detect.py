import os
import csv
import time
import json
from datetime import datetime
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from src.model import create_vit_model, load_checkpoint


class Detector:
    def __init__(self, config_path='config.json', checkpoint=None):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)
        self.device = self.cfg['detection'].get('device', 'cpu')
        self.yolo_model_path = self.cfg['detection'].get('yolo_model')
        # load yolov8 model (user may supply custom trained model that finds sign candidates)
        try:
            self.yolo = YOLO(self.yolo_model_path)
        except Exception:
            self.yolo = None
        self.conf_thres = self.cfg['detection'].get('confidence_threshold', 0.5)

        # load ViT
        # number of classes is unknown here; will be set by user after initializing with classes length
        self.vit = None
        self.vit_checkpoint = checkpoint

        self.sign_meanings = {}
        try:
            with open('sign_meanings.json','r') as f:
                self.sign_meanings = json.load(f)
        except Exception:
            self.sign_meanings = {}

        os.makedirs(os.path.dirname(self.cfg['paths']['logs']), exist_ok=True)

    def load_vit(self, num_classes):
        self.vit = create_vit_model(num_classes, image_size=self.cfg['training']['image_size'], pretrained=False)
        if self.vit_checkpoint:
            load_checkpoint(self.vit_checkpoint, model=self.vit, map_location=self.device)
        self.vit.to(self.device)
        self.vit.eval()

    def classify_crop(self, crop):
        # crop: numpy RGB
        import torchvision.transforms as T
        from PIL import Image
        img = Image.fromarray(crop)
        tf = T.Compose([T.Resize(self.cfg['training']['image_size']), T.CenterCrop(self.cfg['training']['image_size']), T.ToTensor(), T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))])
        x = tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.vit(x)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            idx = int(probs.argmax())
            conf = float(probs[idx])
        return idx, conf, probs

    def detect_frame(self, frame, classes):
        # frame: BGR numpy
        h, w = frame.shape[:2]
        detections = []
        # 1) use YOLO to get candidate boxes
        if self.yolo is not None:
            try:
                results = self.yolo(frame[..., ::-1])
                for res in results:
                    for box in res.boxes:
                        conf = float(box.conf[0])
                        if conf < self.conf_thres:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        # crop and classify
                        crop = frame[y1:y2, x1:x2][:, :, ::-1]  # BGR->RGB
                        if crop.size == 0:
                            continue
                        cls_idx, cls_conf, probs = self.classify_crop(crop)
                        cls_name = classes[cls_idx]
                        meaning = self.sign_meanings.get(cls_name, {})
                        detections.append({
                            'box': (x1, y1, x2, y2),
                            'class': cls_name,
                            'confidence': cls_conf,
                            'meaning_en': meaning.get('en',''),
                            'meaning_ta': meaning.get('ta',''),
                            'instruction': meaning.get('instruction','')
                        })
            except Exception:
                pass

        # fallback: sliding window / coarse grid (simple) when YOLO unavailable
        if self.yolo is None:
            # perform a simple grid of crops
            step = max(32, w//6)
            size = min(w,h)//4
            for y in range(0, h-size, step):
                for x in range(0, w-size, step):
                    crop = frame[y:y+size, x:x+size][:,:,::-1]
                    cls_idx, cls_conf, probs = self.classify_crop(crop)
                    if cls_conf > self.conf_thres:
                        cls_name = classes[cls_idx]
                        meaning = self.sign_meanings.get(cls_name, {})
                        detections.append({
                            'box': (x, y, x+size, y+size),
                            'class': cls_name,
                            'confidence': cls_conf,
                            'meaning_en': meaning.get('en',''),
                            'meaning_ta': meaning.get('ta',''),
                            'instruction': meaning.get('instruction','')
                        })

        # log detections
        for d in detections:
            self.log_detection(d)

        return detections

    def log_detection(self, det):
        row = [datetime.utcnow().isoformat(), det['class'], det['confidence'], det['box'][0], det['box'][1], det['box'][2], det['box'][3], det.get('meaning_en',''), det.get('meaning_ta',''), det.get('instruction','')]
        with open(self.cfg['paths']['logs'], 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)


if __name__ == '__main__':
    # simple test: open webcam and run detection with a dummy class list
    det = Detector()
    det.load_vit(num_classes=5)
    cap = cv2.VideoCapture(0)
    classes = ['speed_limit_20','stop','yield','no_entry','turn_right']
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        outs = det.detect_frame(frame, classes)
        for o in outs:
            x1,y1,x2,y2 = o['box']
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame, f"{o['class']}:{o['confidence']:.2f}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
        cv2.imshow('det', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
