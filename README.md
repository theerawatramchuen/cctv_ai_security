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

  * __The original frame (as JPEG)

  * __The annotated frame (with bounding boxes)

  * __A 120‑second video clip centred on the detection (60 seconds before, 60 seconds after)

* __Resume after clip –__ after saving a clip, the processing jumps forward to just after the clip ends to avoid duplicate triggers.

* __Live display –__ shows the video with real‑time annotations and performance metrics. Press space to pause, q to quit.
