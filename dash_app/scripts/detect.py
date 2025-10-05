"""Real-time detection: OpenCV webcam reader, run a detector (YOLO if available) and a classifier (ViT) for each crop.
Logs detections to logs/detections.csv
"""
import cv2
import time
import os
import csv
import torch
from scripts.model import create_vit_model
from torchvision import transforms
import numpy as np


def init_video_source(source=0):
    cap = cv2.VideoCapture(source)
    return cap


def detect_and_classify(frame, classifier, classes, device='cpu', conf_threshold=0.5):
    # Placeholder: use simple color-based contour detection as a stand-in for YOLO
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h < 500: continue
        crop = frame[y:y+h, x:x+w]
        inp = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            out = classifier(inp)
            probs = torch.nn.functional.softmax(out, dim=1)
            conf, idx = torch.max(probs, dim=1)
            conf = conf.item(); idx = idx.item()
        if conf >= conf_threshold:
            detections.append({'box':(x,y,w,h),'class':classes[idx],'conf':float(conf)})
    return detections


def stream_webcam(classifier, classes, device='cpu', conf_threshold=0.6):
    cap = init_video_source(0)
    os.makedirs('logs', exist_ok=True)
    csvpath = os.path.join('logs','detections.csv')
    if not os.path.exists(csvpath):
        with open(csvpath,'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp','class','confidence'])
    while True:
        ret, frame = cap.read()
        if not ret: break
        dets = detect_and_classify(frame, classifier, classes, device, conf_threshold)
        for d in dets:
            x,y,w,h = d['box']
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame, f"{d['class']} {d['conf']:.2f}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0),2)
            with open(csvpath,'a',newline='') as f:
                writer = csv.writer(f)
                writer.writerow([time.time(), d['class'], d['conf']])
        cv2.imshow('TSR', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # quick demo loader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load a ViT model placeholder (must match your training classes)
    model = create_vit_model(43)
    model.eval()
    stream_webcam(model, [str(i) for i in range(43)], device)
