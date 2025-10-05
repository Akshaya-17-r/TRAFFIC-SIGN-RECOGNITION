"""
Simple wrapper that tries to load YOLO (ultralytics) and a ViT classifier (tsr_project).
Provides a detect(frame) method returning list of detections: {label, conf, bbox}.
If YOLO is unavailable, falls back to a lightweight contour detector.
"""
import os
import time
import json
import numpy as np
import cv2

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False

try:
    from tsr_project.model_utils import create_model, load_checkpoint
    import torch
    _HAS_VIT = True
except Exception:
    _HAS_VIT = False


class HybridDetector:
    def __init__(self, cfg_path=None, device='cpu'):
        self.device = device
        self.yolo = None
        self.vit = None
        self.classes = None
        self.yolo_class_map = None
        cfg = {}
        if cfg_path and os.path.exists(cfg_path):
            try:
                cfg = json.load(open(cfg_path))
            except Exception:
                cfg = {}

        if _HAS_YOLO:
            yolo_path = cfg.get('detection', {}).get('yolo_model', 'yolov8n.pt')
            try:
                self.yolo = YOLO(yolo_path)
                # try to load class names from the YOLO model if available
                try:
                    # ultralytics YOLO object may expose model.names
                    self.yolo_class_map = getattr(self.yolo.model, 'names', None) or getattr(self.yolo, 'names', None)
                except Exception:
                    self.yolo_class_map = None
            except Exception:
                self.yolo = None

        # Try to load ViT final checkpoint from tsr_project checkpoints/final.pth
        if _HAS_VIT:
            try:
                # infer classes from dataset folder if present
                train_dir = cfg.get('dataset', {}).get('train_dir')
                if train_dir and os.path.exists(train_dir):
                    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir,d))])
                    self.classes = classes
                    self.vit = create_model(len(classes), device=device)
                    ckpt = cfg.get('paths', {}).get('checkpoints','checkpoints')
                    final_path = os.path.join(ckpt, 'final.pth')
                    if os.path.exists(final_path):
                        load_checkpoint(self.vit, final_path, device=device)
                else:
                    self.vit = None
            except Exception:
                self.vit = None

    def detect(self, frame, conf_threshold=0.4):
        """Run detection and optionally classify crops with ViT.

        Returns list of dicts: {label, conf, bbox}
        """
        detections = []
        h, w = frame.shape[:2]
        if self.yolo is not None:
            try:
                results = self.yolo(frame)[0]
                boxes = results.boxes
                for b, c, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                    if c < conf_threshold:
                        continue
                    xmin, ymin, xmax, ymax = map(int, b)
                    # try to get readable label from YOLO mapping
                    try:
                        if self.yolo_class_map is not None:
                            label = str(self.yolo_class_map[int(cls)])
                        else:
                            label = str(int(cls))
                    except Exception:
                        label = str(int(cls))
                    conf = float(c)
                    # reclassify with ViT if available
                    if self.vit is not None:
                        crop = frame[max(0,ymin):min(h,ymax), max(0,xmin):min(w,xmax)]
                        if crop.size == 0:
                            continue
                        crop_resized = cv2.resize(crop, (224,224))
                        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                        tensor = (np.asarray(crop_rgb)/255.0).astype('float32')
                        tensor = np.transpose(tensor, (2,0,1))
                        import torch
                        tens = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            out = self.vit(tens)
                            probs = torch.softmax(out, dim=1)
                            pconf, pred = torch.max(probs, dim=1)
                            # prefer human-readable classes from ViT
                            if self.classes is not None:
                                label = str(self.classes[int(pred.cpu().item())])
                            else:
                                label = str(int(pred.cpu().item()))
                            conf = float(pconf.cpu().item())
                    detections.append({'label':label, 'conf':conf, 'bbox':(xmin,ymin,xmax,ymax)})
                return detections
            except Exception:
                # fall back
                pass

        # YOLO not available: use simple contour detection fallback
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x,y,wc,hc = cv2.boundingRect(cnt)
            if wc*hc < 1000:
                continue
            label = 'unknown'
            conf = 0.5
            # attempt ViT classification
            if self.vit is not None and self.classes is not None:
                crop = frame[y:y+hc, x:x+wc]
                crop_resized = cv2.resize(crop, (224,224))
                crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                tensor = (np.asarray(crop_rgb)/255.0).astype('float32')
                tensor = np.transpose(tensor, (2,0,1))
                import torch
                tens = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out = self.vit(tens)
                    probs = torch.softmax(out, dim=1)
                    pconf, pred = torch.max(probs, dim=1)
                    label = self.classes[int(pred.cpu().item())]
                    conf = float(pconf.cpu().item())
            detections.append({'label':label, 'conf':conf, 'bbox':(x,y,x+wc,y+hc)})
        return detections
