"""YOLO + ViT hybrid detection used by Dash real-time page."""
import os
import json
import torch
import cv2
from ultralytics import YOLO
from pathlib import Path
from src.model import create_vit_model, load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / 'sign_meanings.json','r') as f:
    SIGN_MEANINGS = json.load(f)


class RealtimeDetector:
    def __init__(self, yolo_path=None, vit_ckpt=None, device='cpu'):
        self.device = device
        self.yolo = None
        if yolo_path and os.path.exists(yolo_path):
            self.yolo = YOLO(yolo_path)
        self.vit = None
        self.vit_ckpt = vit_ckpt

    def load_vit(self, num_classes):
        self.vit = create_vit_model(num_classes)
        if self.vit_ckpt:
            load_checkpoint(self.vit_ckpt, model=self.vit)
        self.vit.to(self.device)
        self.vit.eval()

    def detect_frame(self, frame, classes, conf_th=0.5):
        # frame BGR
        outs = []
        if self.yolo:
            res = self.yolo(frame[...,::-1])
            for r in res:
                for b in r.boxes:
                    conf = float(b.conf[0])
                    if conf < conf_th: continue
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    crop = frame[y1:y2, x1:x2][:,:,::-1]
                    # classify with vit if available
                    if self.vit:
                        import torchvision.transforms as T
                        from PIL import Image
                        img = Image.fromarray(crop)
                        tf = T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))])
                        x = tf(img).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            out = self.vit(x)
                            idx = int(out.argmax(dim=1).cpu().numpy()[0])
                            score = float(torch.softmax(out, dim=1)[0,idx].cpu().numpy())
                        cls = classes[idx]
                    else:
                        cls = 'unknown'
                        score = conf
                    outs.append({'box':(x1,y1,x2,y2),'class':cls,'confidence':score})
        return outs
