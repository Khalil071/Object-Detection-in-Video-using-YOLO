Object Detection in Video using YOLO

Overview

This project builds an object detection model using YOLO (You Only Look Once) to detect objects in video streams. YOLO is a fast and accurate deep learning model that can identify multiple objects in real-time, making it ideal for video processing applications.

Features

Real-time object detection in video streams

Use of pre-trained YOLO model (e.g., YOLOv5, YOLOv8)

Bounding box visualization for detected objects

Support for custom object detection by training on a new dataset

Ability to process both stored videos and live camera feeds

Technologies Used

Python

YOLO (You Only Look Once)

OpenCV

PyTorch

TensorFlow/Keras (optional)

Matplotlib/Seaborn

Installation

Clone the repository:

git clone https://github.com/yourusername/object-detection-video-yolo.git
cd object-detection-video-yolo

Install dependencies:

pip install -r requirements.txt

Install YOLOv5 (if using YOLOv5):

git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt

Model Usage

Detect Objects in a Video File

Load the YOLO model:

from ultralytics import YOLO
import cv2

model = YOLO("yolov5s.pt")  # Load the YOLOv5 small model

Process a video file:

cap = cv2.VideoCapture("path/to/video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    results.show()  # Display video with detected objects
cap.release()

Real-time Object Detection using Webcam

Capture live video:

cap = cv2.VideoCapture(0)  # Open webcam
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    results.show()
cap.release()

Future Improvements

Optimize YOLO for real-time performance on edge devices

Train YOLO on a custom dataset for specific object detection

Deploy as a web-based or mobile application

Implement multi-camera object tracking

License

This project is licensed under the MIT License.
