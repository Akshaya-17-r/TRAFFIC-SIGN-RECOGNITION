Traffic Sign Recognition Dashboard (Dash + ViT) 🚦

Traffic Sign Recognition (TSR) – Mini Driver Assistance Prototype

This project is a Traffic Sign Recognition system designed as a mini driver assistance prototype. It combines a Vision Transformer (ViT) model for traffic sign classification with a YOLO-based detector for detecting multiple signs in real-time. The system also provides voice alerts, a dashboard interface, and logging for safety-critical applications.

Overview

Road safety can be enhanced with automated traffic sign recognition. This project demonstrates a pipeline that:

Processes traffic sign images from the GTSRB dataset.

Trains a Vision Transformer (ViT) for accurate classification.

Integrates YOLO-based detection to detect multiple signs in real-time camera feeds.

Provides a dashboard (Dash & Streamlit) for live visualization.

Issues voice alerts in English and Tamil for driver awareness.

The system can be extended for driver assistance systems (ADAS) and educational purposes.

Architecture

Data Preprocessing: Images are resized, normalized, and augmented to improve model generalization.

Vision Transformer (ViT): Pretrained on ImageNet, fine-tuned on traffic sign classes for high accuracy.

YOLO Detector: Detects multiple signs simultaneously in a frame, enabling hybrid detection.

Real-Time Detection: Uses OpenCV to process video streams, overlay detection results, and trigger alerts.

Dashboard & Logging:

Dash dashboard provides a real-time interactive interface.

Streamlit app supports live detection visualization.

Logs detection events for analysis.

Key Features

Accurate classification and detection of traffic signs.

Multi-sign detection in a single frame.

Real-time alerts for drivers to enhance safety.

Visual dashboards for monitoring and demonstration.

Supports English and Tamil voice notifications.

Use Cases

Driver Assistance: Alerts drivers about upcoming traffic signs.

Traffic Monitoring: Analyze road signs in video streams.

Educational Tool: Demonstrates the integration of computer vision models in real-world scenarios.

Project Structure

src/ – Core code for training, detection, and dashboard.

tsr_project/ – Python package for model pipeline, detection, and dashboards.

dash_app/ – Dash dashboard implementation.

dataset/ – Contains GTSRB dataset (train/test folders).

checkpoints/ – Stores trained model weights.

results/ – Evaluation outputs and misclassified images.

logs/ – Detection logs in CSV format.

config.json – Runtime configurations for dataset paths, thresholds, and hyperparameters.

sign_meanings.json – Mapping of class IDs to traffic sign meanings and instructions
