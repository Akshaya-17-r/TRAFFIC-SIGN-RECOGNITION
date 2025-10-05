# Traffic Sign Recognition Dashboard (Dash + ViT)

This project is a prototype Traffic Sign Recognition system with a Dash dashboard, ViT model, preprocessing, training, evaluation and real-time detection.

Place the GTSRB dataset under `dataset/train` and `dataset/test` as directories of class folders. Install requirements and run the dashboard:

```
pip install -r dash_app/requirements.txt
python dash_app/app.py
```

See individual scripts in `dash_app/scripts/` for training, preprocessing, evaluation and detection.
# Traffic Sign Recognition (TSR) - Mini Driver Assistance Prototype

This project provides a full Traffic Sign Recognition pipeline with training,
evaluation, a YOLO+ViT hybrid detection pipeline, voice alerts, and a Streamlit
dashboard for live detection.

Place the GTSRB dataset under `dataset/train` and `dataset/test` as class folders.

Setup

1. Create a virtual environment and activate it.
2. Install dependencies:

```powershell
# from workspace root (d:\traffic)
python -m pip install -r requirements.txt
```

Training

```powershell
python -m tsr_project.train
```

Run real-time detection

```powershell
python -m tsr_project.detect
```

Run Streamlit dashboard

```powershell
streamlit run tsr_project/dashboard.py
```

Files

- `tsr_project/` - main package
- `config.json` - configuration for dataset paths and hyperparams
- `requirements.txt` - python deps
- `sign_meanings.json` - mapping of class -> meaning and instructionTraffic Sign Recognition (ViT + YOLO Hybrid)

Overview
--------
This project implements a Traffic Sign Recognition system using a Vision Transformer (ViT) pretrained on ImageNet combined with a YOLO-based detector for multi-sign detection. It includes training, evaluation, a real-time OpenCV+Streamlit dashboard, voice alerts (English + Tamil), and logging.

Place the GTSRB dataset under:

  dataset/train
  dataset/test


Install dependencies (recommended inside a virtualenv):

Windows PowerShell example:

```powershell
cd d:\traffic
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Quick start (run dashboard):

```powershell
streamlit run src/streamlit_app.py
```

Run training (example):

```powershell
python run.py --mode train --config config.json
```

Run evaluation (example):

```powershell
python run.py --mode eval --config config.json --checkpoint checkpoints/best.pt
```

Run a quick smoke test to validate imports and model forward pass:

```powershell
python test_smoke.py
```


Project structure
-----------------
- src/: main source code
- dataset/: expected dataset location (not included)
- checkpoints/: model checkpoints saved during training
- results/: evaluation outputs and misclassified images
- logs/: detection CSV logs
- config.json: runtime configuration

See `config.json` to configure paths, training hyperparameters, and detection thresholds.

Run & Deploy (Dash)
-------------------
From the workspace root (PowerShell):

```powershell
# create venv and activate
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# run the Dash dashboard
python .\dash_app\app.py
```

Optional: build a Docker image (example): create a Dockerfile that installs requirements and runs `python dash_app/app.py` and then build and run.
