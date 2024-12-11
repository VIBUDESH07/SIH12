
# Face Liveness Detection and Anti-Spoofing System

This project implements a face liveness detection system using a combination of deep learning techniques, including a YOLOv5 trained model, a Convolutional Neural Network (CNN), and ONNX for model inference. The system also integrates virtual camera prevention and a deepfake detection model based on FaceForensics++.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Virtual Camera Prevention](#virtual-camera-prevention)
- [Deepfake Detection](#deepfake-detection)
- [References](#references)

## Introduction
Face liveness detection is crucial for preventing spoofing attacks in facial recognition systems. This project uses multiple models to enhance robustness against attacks, including the use of printed images, videos, and deepfakes.

## Features
- **YOLOv5 for Face Detection**: Utilizes a trained YOLOv5 model to accurately detect faces in real-time.
- **CNN for Liveness Detection**: A custom CNN model distinguishes between live and spoofed faces.
- **ONNX Inference**: ONNX runtime enables fast and efficient inference across platforms.
- **Virtual Camera Prevention**: The system detects and blocks inputs from virtual cameras to prevent replay attacks.
- **Deepfake Detection**: Integrates a deepfake detection model trained with the FaceForensics++ dataset.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face-liveness-detection.git
   cd face-liveness-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the required models:
   - Download the YOLOv5 trained model.
   - Download the CNN model for liveness detection.
   - Download the FaceForensics++ model for deepfake detection.

4. Install ONNX runtime for model inference:
   ```bash
   pip install onnxruntime
   ```

5. (Optional) Set up your virtual environment if required.

## Usage

1. Run the application:
   ```bash
   python app.py
   ```

2. Webcam input will be captured for real-time face detection and liveness analysis.

3. The system will:
   - Detect if the input is from a virtual camera and prevent it.
   - Determine if the face is live or spoofed.
   - Run deepfake detection on the detected face.

## Model Details

### YOLOv5 Model
The YOLOv5 model is trained for precise face detection in real-time. This model ensures fast detection with minimal latency.

### CNN Liveness Detection Model
This CNN model distinguishes between live and spoofed faces by analyzing facial textures and patterns that cannot be replicated with 2D images or videos.

### ONNX Runtime
ONNX enables efficient model inference, allowing us to run the models seamlessly across different devices and platforms.

## Virtual Camera Prevention
To enhance security, the system includes functionality to detect and block the use of virtual cameras, which are often used in replay attacks.

## Deepfake Detection
A deepfake detection model, trained on the FaceForensics++ dataset, is integrated to analyze and detect faces that may be generated or manipulated by AI.

## References
- YOLOv5: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- FaceForensics++: [https://github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)
- ONNX Runtime: [https://onnxruntime.ai/](https://onnxruntime.ai/)
