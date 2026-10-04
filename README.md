# 🏟️ Sports Player Detection using YOLO11

Detects players and the ball in sports images and video using the pre-trained
Ultralytics YOLO11n model. No model training was done; the project uses
inference on a model already trained on the COCO dataset.

## 🔗 Links
- **Live demo:** https://sports-player-detection-suma.streamlit.app
- **GitHub repository:** https://github.com/Suma-2005/sports-player-detection
- **Model source:** https://github.com/ultralytics/ultralytics
- **COCO dataset:** https://cocodataset.org

## Pipeline
Image/video → YOLO11n (pre-trained) → boxes, classes, confidence → filter person and sports ball → count and track players → annotated output

## Features
- Upload a sports image and get players and balls detected with boxes
- Adjustable confidence threshold
- Player and ball count
- Video tracking demo with an ID for each player (`output.mp4`)

## Results (sample video, 300 frames)
- Average players per frame: 1.6
- Maximum players in one frame: 2
- Unique IDs assigned: 5 (extra IDs come from ID switches after occlusion)

## How it works
YOLO ("You Only Look Once") passes the whole image through the network once,
which makes it fast enough for video. For every object it outputs a bounding
box, a class and a confidence score. Detections below the confidence
threshold (0.4) are removed, and non-maximum suppression (NMS) removes
duplicate boxes. Only the classes person (0) and sports ball (32) are kept.

## Limitations
- Cannot tell teams or referees apart
- Small, fast objects like the ball are sometimes missed
- Tracker IDs can switch when players overlap or leave the frame

## Source and license
- Model: Ultralytics YOLO11, https://github.com/ultralytics/ultralytics
- Weights: yolo11n.pt, pre-trained on the COCO dataset
- License: AGPL-3.0 (please confirm on the LICENSE page of the Ultralytics repository)

## Run locally
pip install -r requirements.txt
python -m streamlit run app.py

## Tech
Python, Ultralytics YOLO11, OpenCV, Streamlit, Google Colab
