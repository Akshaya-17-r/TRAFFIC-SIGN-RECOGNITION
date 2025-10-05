import os
import csv
import pandas as pd


def log_detection(timestamp, cls, conf, csvpath='logs/detections.csv'):
    os.makedirs(os.path.dirname(csvpath), exist_ok=True)
    exists = os.path.exists(csvpath)
    with open(csvpath, 'a', newline='') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(['timestamp','class','confidence'])
        writer.writerow([timestamp, cls, conf])


def read_logs(csvpath='logs/detections.csv'):
    if not os.path.exists(csvpath):
        return pd.DataFrame(columns=['timestamp','class','confidence'])
    return pd.read_csv(csvpath)
