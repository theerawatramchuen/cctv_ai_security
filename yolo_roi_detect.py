"""
YOLO ROI Detection — Approach 1 (pre-mask)
==========================================
Loads ROI polygon(s) from the JSON file produced by roi_polygon_editor.py
and runs YOLO inference only inside those regions.

JSON is expected beside the source file with the same base name:
  e.g.  1.mp4  →  1.json

Usage:
  python yolo_roi_detect.py --source 1.mp4
  python yolo_roi_detect.py --source 1.mp4 --json custom_roi.json
  python yolo_roi_detect.py --source 1.jpg
  python yolo_roi_detect.py --source 0              # webcam
  python yolo_roi_detect.py --source 1.mp4 --roi-ids 0 2
  python yolo_roi_detect.py --source 1.mp4 --save out.mp4
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

PALETTE = [
    (0, 255, 120),
    (0, 180, 255),
    (255, 100, 0),
    (200, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Only these COCO class names will be detected and drawn.
# Edit freely — names must match YOLO's model.names (lowercase).
TARGET_CLASSES = {"person", "dog", "cat"}

# Per-class box colours  (BGR)
CLASS_COLORS = {
    "person": (0,  200, 255),   # yellow-orange
    "dog":    (80, 255, 80),    # green
    "cat":    (255, 100, 80),   # blue
}
DEFAULT_COLOR = (200, 200, 200)


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


# ──────────────────────────────────────────────
#  JSON loader
# ──────────────────────────────────────────────

def load_roi_polygons(json_path: str,
                      roi_ids: list | None = None) -> list[np.ndarray]:
    """
    Load ROI polygons from JSON produced by roi_polygon_editor.py.

    Returns list of np.ndarray each shaped (N, 1, 2) int32 — cv2-ready.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"ROI JSON not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    polygons_raw = data.get("polygons", [])
    if not polygons_raw:
        raise ValueError(f"No polygons found in {json_path}")

    result = []
    for poly in polygons_raw:
        pid = poly.get("id", -1)
        if roi_ids is not None and pid not in roi_ids:
            continue
        pts = np.array(poly["points"], dtype=np.int32)  # (N, 2)
        pts = pts.reshape((-1, 1, 2))                    # (N, 1, 2) for cv2
        result.append(pts)

    if not result:
        raise ValueError(f"No matching polygons (ids={roi_ids}) in {json_path}")

    print(f"[roi] Loaded {len(result)} polygon(s) from: {json_path}")
    return result


def discover_json(source_path: str) -> str:
    """Return <same-dir/basename>.json beside the source file."""
    return os.path.splitext(os.path.abspath(source_path))[0] + ".json"


# ──────────────────────────────────────────────
#  Mask builder
# ──────────────────────────────────────────────

def build_roi_mask(frame_shape: tuple,
                   polygons: list[np.ndarray]) -> np.ndarray:
    """
    Build a single-channel binary mask (255 inside ROI, 0 outside).
    polygons: list of (N,1,2) int32 arrays.
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    # cv2.fillPoly wants a list of (N,1,2) arrays — already the right shape
    cv2.fillPoly(mask, polygons, color=255)
    return mask


# ──────────────────────────────────────────────
#  Per-frame inference
# ──────────────────────────────────────────────

def detect_in_roi(frame: np.ndarray,
                  mask: np.ndarray,
                  model: YOLO,
                  conf: float = 0.25):
    """
    1. Apply mask to zero pixels outside ROI.
    2. Run YOLO on masked frame.
    3. Draw boxes on the ORIGINAL frame (clean background).
    Returns (annotated_frame, results).
    """
    # Black out everything outside the ROI
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Run inference
    results = model(masked_frame, conf=conf, verbose=False)

    # Draw detections on original (unmasked) frame — TARGET_CLASSES only
    annotated = frame.copy()
    for box in results[0].boxes:
        cls_id    = int(box.cls[0])
        cls_name  = model.names[cls_id].lower()

        # ── skip anything not in TARGET_CLASSES
        if cls_name not in TARGET_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf_val  = float(box.conf[0])
        label     = f"{cls_name} {conf_val:.2f}"
        color     = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return annotated, results


# ──────────────────────────────────────────────
#  ROI overlay
# ──────────────────────────────────────────────

def draw_roi_overlay(frame: np.ndarray,
                     polygons: list[np.ndarray]) -> None:
    """Draw semi-transparent ROI fills + coloured borders IN PLACE."""
    overlay = frame.copy()
    for i, poly in enumerate(polygons):
        color = PALETTE[i % len(PALETTE)]
        cv2.fillPoly(overlay, [poly], color)
    # blend fill at 15 % opacity
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    for i, poly in enumerate(polygons):
        color = PALETTE[i % len(PALETTE)]
        cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)
        pts = poly.reshape(-1, 2)
        cx  = int(pts[:, 0].mean())
        cy  = int(pts[:, 1].mean())
        cv2.putText(frame, f"ROI {i}", (cx - 22, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"ROI {i}", (cx - 22, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="YOLO ROI detection — pre-mask approach")
    ap.add_argument("--source",    default="0",
                    help="Video/image path or camera index (default: 0)")
    ap.add_argument("--json",      default=None,
                    help="ROI JSON path. Auto-discovered from --source if omitted.")
    ap.add_argument("--model",     default="yolo11s.pt",
                    help="YOLO weights (default: yolo11s.pt)")
    ap.add_argument("--conf",      type=float, default=0.25,
                    help="Confidence threshold (default: 0.25)")
    ap.add_argument("--roi-ids",   type=int, nargs="*", default=None,
                    help="Polygon IDs to use (default: all)")
    ap.add_argument("--no-overlay", action="store_true",
                    help="Hide ROI boundary overlay")
    ap.add_argument("--save",      default=None,
                    help="Save output to this video path (e.g. out.mp4)")
    args = ap.parse_args()

    # ── resolve source type
    is_camera = args.source.isdigit()
    source    = int(args.source) if is_camera else args.source

    # ── resolve JSON
    json_path = args.json
    if json_path is None:
        if is_camera:
            print("[error] --json is required when source is a camera index.")
            sys.exit(1)
        json_path = discover_json(args.source)

    # ── load polygons
    try:
        polygons = load_roi_polygons(json_path, roi_ids=args.roi_ids)
    except (FileNotFoundError, ValueError) as e:
        print(f"[error] {e}")
        sys.exit(1)

    # ── load model
    print(f"[model] Loading {args.model} …")
    model = YOLO(args.model)
    print(f"[model] Ready.")

    # ── open video/image/camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[error] Cannot open source: {args.source}")
        sys.exit(1)

    # ── optional video writer
    writer = None
    if args.save:
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
        print(f"[save] Writing → {args.save}")

    mask        = None   # built on first frame
    frame_count = 0
    static_img  = not is_camera and is_image(str(source))

    print("Press  Q / Esc  to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            # end of video — loop back to start (comment out if not desired)
            if not static_img and not is_camera:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                break

        frame_count += 1

        # Build / rebuild mask if resolution changed
        if mask is None or mask.shape[:2] != frame.shape[:2]:
            mask = build_roi_mask(frame.shape, polygons)
            print(f"[mask] Built {mask.shape[1]}×{mask.shape[0]} mask "
                  f"for {len(polygons)} polygon(s).")

        # ── inference
        annotated, results = detect_in_roi(frame, mask, model, conf=args.conf)

        # ── ROI overlay (in-place on annotated)
        if not args.no_overlay:
            draw_roi_overlay(annotated, polygons)

        # ── frame counter HUD
        cv2.putText(annotated, f"Frame: {frame_count}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # ── show
        cv2.imshow("YOLO ROI Detection", annotated)

        # ── optional save
        if writer:
            writer.write(annotated)

        # ── key handling
        wait_ms = 0 if static_img else 1        # 0 = wait forever for images
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord('q'), ord('Q'), 27):     # Q or Esc
            break

        # static image: show until any key pressed, then exit
        if static_img:
            break

    cap.release()
    if writer:
        writer.release()
        print(f"[save] Saved → {args.save}")
    cv2.destroyAllWindows()
    print(f"[done] Processed {frame_count} frame(s).")


if __name__ == "__main__":
    main()
