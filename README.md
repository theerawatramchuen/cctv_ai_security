# cctv_ai_security
This Python script integrates three different security video analysis use cases into a single tool:

* __Scrap area:__ Detects when a person overlaps with a car or truck.

* __Wirebond area:__ Detects when a vacuume object is inside or overlapping a normal or suspected area, or when a spool is inside/overlapping a grove.

* __Store area:__ Detects any person present in the frame.

The script processes all .mp4 files in a given folder, runs YOLO inference on each frame, and saves annotated images and 120‑second video clips when a condition is __validated__ (i.e., persists for a user‑defined number of seconds).

This Python script is base on original repsitory https://github.com/theerawatramchuen/find_vacuumeNspool_holding_clip

# Features
* __Single entry point –__ choose the analysis mode via a command‑line argument.

* __Per‑area configuration –__ each area uses its own condition logic and video folder.

* __Validation time –__ prevents false positives by requiring a condition to be stable for a specified duration.

* __Clip saving –__ when a condition is validated, the script saves:

  * The original frame (as JPEG)

  * The annotated frame (with bounding boxes)

  * A 120‑second video clip centred on the detection (60 seconds before, 60 seconds after)

* __Resume after clip –__ after saving a clip, the processing jumps forward to just after the clip ends to avoid duplicate triggers.

* __Live display –__ shows the video with real‑time annotations and performance metrics. Press space to pause, q to quit.

# Requirements
* Python 3.8+

* Ultralytics YOLO

* OpenCV (cv2)

* NumPy

Install dependencies with:
```
pip install ultralytics opencv-python numpy
```
# Configuration
Before running, you must set the paths to your video folders.
Edit the constants at the top of the script:
```
SCRAP_VIDEO_FOLDER = r"D:\cctvdownload\_scrap"        # folder with videos for scrap area
WIREBOND_VIDEO_FOLDER = r"D:\cctvdownload\_wirebond"      # folder with videos for wirebond area
STORE_VIDEO_FOLDER = r"D:\cctvdownload\_store"       # folder with videos for store area
```
Each folder should contain one or more .mp4 files. The script processes them sequentially.

# Usage
Run the script from the command line:
```
python combined_security.py <area> <model_path> [options]
```
# Positional Arguments
| Argument      |	Description                                                     | 
| ------------- |:---------------------------------------------------------------:|
| area          | One of: scrap, wirebond, store – selects the detection logic.   |
| model_path    | Path to the YOLO model weights file (e.g. yolov8n.pt, best.pt). |
## Optional Arguments

