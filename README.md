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
| `area`        | One of: `scrap`, `wirebond`, `store` – selects the detection logic. |
| `model_path`  | Path to the YOLO model weights file (e.g. `yolov8n.pt`, `best.pt`). |

# Optional Arguments
| Option	            | Default	   | Description                                        |
|--------------------|:----------:|----------------------------------------------------|
|`--conf`            |`0.5`       |Confidence threshold for detections. Only detections above this are considered.|
|`--validation_time` |`2.0`       |Time in seconds a condition must persist before being considered valid.|
|`--line_thickness`  |`1`         |Thickness of bounding boxes drawn on the annotated frames.|

# Examples
1. __Scrap area__ with a COCO‑pretrained model:
```
python combined_security.py scrap C:\models\yolo11s.pt
```
2. __Wirebond area__ with a custom‑trained model, higher confidence, shorter validation:
```
python combined_security.py wirebond D:\weights\best.pt --conf 0.7 --validation_time 1.0
```
3. __Store area__ with a person‑detection model:
```
python combined_security.py store /home/user/models/person.pt
```
# Output
For each video, the script creates an output folder named after the area (e.g. `scrap_area_output`, `store_area_output`). Inside, you will find:

* __Images:__

 * `YYYYMMDD-HHMMSS-XXX-O.jpg` – original frame at the moment of validation.

 * `YYYYMMDD-HHMMSS-XXX-R.jpg` – same frame with YOLO annotations.

* __Video clips:__

 * `YYYYMMDD-HHMMSS-XXX.mp4` – a 120‑second clip centred on the detection (60 s before, 60 s after).

The timestamp is generated when the condition is first validated; the XXX is a millisecond value to ensure uniqueness.

# How It Works
1. The script scans the chosen video folder for all .mp4 files.
2. For each video, it reads frames and runs the YOLO model.
3. Detections are grouped by class and filtered by confidence.
4. Depending on the selected area, the script checks for specific conditions:
 * Scrap: person overlapping car or truck.
 * Wirebond: vacuume inside/overlapping normal or suspected; spool inside/overlapping grove.
 * Store: any person detection.
5. Each unique condition (based on object positions) is tracked across frames.
If it persists for at least validation_time seconds, it is validated.
6. Upon validation, the script saves the two images and a 120‑second video clip.
It then jumps forward in the video to just after the clip ends to avoid repeated triggers.
7.The live display shows the current frame, processing speed, and video information.
