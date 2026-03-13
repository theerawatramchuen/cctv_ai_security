# cctv_ai_security
This Python script integrates three different security video analysis use cases into a single tool:

Scrap area: Detects when a person overlaps with a car or truck.

Wirebond area: Detects when a vacuume object is inside or overlapping a normal or suspected area, or when a spool is inside/overlapping a grove.

Store area: Detects any person present in the frame.

The script processes all .mp4 files in a given folder, runs YOLO inference on each frame, and saves annotated images and 120‑second video clips when a condition is validated (i.e., persists for a user‑defined number of seconds).
