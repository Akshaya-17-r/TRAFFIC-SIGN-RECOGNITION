Traffic Sign Recognition Dashboard (Dash + ViT)

Overview
--------
This Dash application provides a full pipeline for Traffic Sign Recognition using PyTorch and Vision Transformer (ViT). It includes dataset preprocessing, training, evaluation, and a real-time detection page.

Place your GTSRB dataset under:

  dataset/train
  dataset/test

Install dependencies (recommended inside a virtualenv):

  pip install -r requirements.txt

Run the Dash app:

  python app.py

Project structure
-----------------
- app.py: Dash app entry point
- assets/: CSS, images, fonts
- data/: dataset utilities
- models/: model definitions and training scripts
- utils/: helpers for preprocessing, audio, logging
- sign_meanings.json: mapping for signs
