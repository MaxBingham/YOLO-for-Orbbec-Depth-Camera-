# Pose Estimation

Real-time human pose estimation from a webcam (or video file) using a YOLO
pose model. The program first **detects people**, draws a **bounding box**
around each one, and then overlays the estimated **skeleton**.

## Setup

```powershell
# from the project root
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
.\venv\Scripts\Activate.ps1
python poseest.py\
```

Optional arguments:

```powershell
python poseest.py\ --source 0            # webcam index (default)
python poseest.py\ --source video.mp4    # a video file
python poseest.py\ --model yolov8s-pose.pt --conf 0.6
```

The model weights (`yolov8n-pose.pt`) download automatically on first run.

## Controls

| Key     | Action                     |
| ------- | -------------------------- |
| `q`/ESC | Quit                       |
| `b`     | Toggle bounding boxes      |
| `s`     | Toggle skeleton drawing    |
| `SPACE` | Pause / resume the stream  |
