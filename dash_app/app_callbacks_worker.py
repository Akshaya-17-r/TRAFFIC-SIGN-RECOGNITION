import threading
import json
import os
from pathlib import Path
from tsr_project.train import train


PROGRESS_PATH = Path(__file__).resolve().parents[1] / 'tmp' / 'train_progress.json'
PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)


def _progress_writer(**kwargs):
    # append epoch progress to JSON file
    p = PROGRESS_PATH
    if p.exists():
        data = json.loads(p.read_text())
    else:
        data = {'epochs': []}
    data['epochs'].append(kwargs)
    p.write_text(json.dumps(data))


def start_training_in_background(config_path='config.json'):
    # clear old progress
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()

    def _worker():
        train(config_path=config_path, progress_callback=_progress_writer)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return True
