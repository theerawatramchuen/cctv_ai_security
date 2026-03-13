import cv2
import os
from datetime import datetime
import time
import logging
import argparse
import glob
import numpy as np
from ultralytics import YOLO

# ----------------------------------------------------------------------
# CONFIGURATION – adjust these paths to match your environment
SCRAP_VIDEO_FOLDER = r"D:\cctvdownload\_scrap"        # folder with videos for scrap area
WIREBOND_VIDEO_FOLDER = r"D:\cctvdownload\_wirebond"      # folder with videos for wirebond area
STORE_VIDEO_FOLDER = r"D:\cctvdownload\_store"       # folder with videos for store area  # <-- NEW
# ----------------------------------------------------------------------

class CombinedVideoYOLOInference:
    def __init__(self, area, model_path, video_folder,
                 confidence_threshold=0.5, line_thickness=2, validation_time=2.0):
        """
        area : str – "scrap", "wirebond", or "store"
        model_path : str – path to YOLO weights
        video_folder : str – directory containing .mp4 files
        """
        self.logger = logging.getLogger(__name__)
        self.area = area.lower()
        self.model_path = model_path
        self.video_folder = video_folder
        self.confidence_threshold = confidence_threshold
        self.line_thickness = line_thickness
        self.validation_time = validation_time
        self.current_video_index = 0
        self.save_dir = f"{self.area}_area_output"   # e.g. scrap_area_output, store_area_output
        self.running = True

        # Condition tracking
        self.condition_frames = {}      # first frame where condition appeared
        self.validated_conditions = set()

        # Video clip settings
        self.clip_target_duration = 120   # total clip length (seconds)
        self.clip_before_duration = 60    # seconds before detection
        self.clip_after_duration = 60     # seconds after detection

        # Load YOLO model
        self.model = YOLO(model_path)

        # Find all mp4 files in the video folder
        self.video_files = glob.glob(os.path.join(video_folder, "*.mp4"))
        if not self.video_files:
            raise ValueError(f"No .mp4 files found in folder: {video_folder}")

        self.logger.info(f"Area: {self.area}")
        self.logger.info(f"Loaded {len(self.video_files)} video files")
        self.logger.info(f"Model classes: {self.model.names}")
        self.logger.info(f"Confidence threshold: {self.confidence_threshold}")
        self.logger.info(f"Validation time: {validation_time} seconds")

    def generate_filename(self, image_type):
        """Generate a filename with timestamp and milliseconds."""
        now = datetime.now()
        milliseconds = int(now.microsecond / 10000)
        return f"{now.strftime('%Y%m%d-%H%M%S')}-{milliseconds}-{image_type}"

    def calculate_iou(self, box1, box2):
        """Intersection over Union of two bounding boxes (x1,y1,x2,y2)."""
        if hasattr(box1, 'cpu'):
            box1 = box1.cpu().numpy()
        if hasattr(box2, 'cpu'):
            box2 = box2.cpu().numpy()
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def boxes_overlap(self, box1, box2):
        """Return True if boxes intersect (IoU > 0). Used for scrap area."""
        return self.calculate_iou(box1, box2) > 0

    def is_inside_or_overlapping(self, inner_box, outer_box, iou_threshold=0.1):
        """
        Used for wirebond area: returns True if inner_box is completely inside
        outer_box OR if they overlap with IoU > iou_threshold.
        """
        if hasattr(inner_box, 'cpu'):
            inner_box = inner_box.cpu().numpy()
        if hasattr(outer_box, 'cpu'):
            outer_box = outer_box.cpu().numpy()

        x1_i, y1_i, x2_i, y2_i = inner_box
        x1_o, y1_o, x2_o, y2_o = outer_box

        # Completely inside?
        if (x1_i >= x1_o and y1_i >= y1_o and x2_i <= x2_o and y2_i <= y2_o):
            return True

        # Overlap with IoU > threshold
        return self.calculate_iou(inner_box, outer_box) > iou_threshold

    def get_condition_key(self, condition_type, det1, det2=None):
        """
        Generate a unique key for a condition involving one or two detections.
        If det2 is None, the key is based only on the first detection.
        """
        bbox1 = det1['bbox'].cpu().numpy() if hasattr(det1['bbox'], 'cpu') else det1['bbox']
        if det2 is not None:
            bbox2 = det2['bbox'].cpu().numpy() if hasattr(det2['bbox'], 'cpu') else det2['bbox']
            # quantise coordinates to group small movements
            return f"{condition_type}_{int(bbox1[0]/10)}_{int(bbox1[1]/10)}_{int(bbox2[0]/10)}_{int(bbox2[1]/10)}"
        else:
            # single detection key (used for store area)
            return f"{condition_type}_{int(bbox1[0]/10)}_{int(bbox1[1]/10)}_{int(bbox1[2]/10)}_{int(bbox1[3]/10)}"

    def save_video_clip(self, video_path, clip_start_frame, filename_base, video_fps, frame_size):
        """Save a 120‑second clip centred on the detection frame."""
        cap = cv2.VideoCapture(video_path)
        rewind = int(self.clip_before_duration * video_fps)
        forward = int(self.clip_after_duration * video_fps)
        total_frames = rewind + forward

        start_frame = max(0, clip_start_frame - rewind)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        clip_path = os.path.join(self.save_dir, f"{filename_base}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_path, fourcc, video_fps, frame_size)

        saved = 0
        try:
            while saved < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                saved += 1
        except Exception as e:
            self.logger.error(f"Error saving video clip: {e}")
        finally:
            out.release()
            cap.release()

        self.logger.info(f"Saved video clip: {clip_path} ({saved} frames)")
        return saved

    # -------------------- Area‑specific condition checks --------------------
    def _check_conditions_scrap(self, detections_by_class):
        """Return a set of condition keys for scrap area (person overlapping car/truck)."""
        current = set()
        person_class = 'person'
        vehicle_classes = {'car', 'truck'}

        if person_class not in detections_by_class:
            return current

        persons = detections_by_class[person_class]
        vehicles = []
        for vcls in vehicle_classes:
            if vcls in detections_by_class:
                vehicles.extend(detections_by_class[vcls])

        for p in persons:
            for v in vehicles:
                if self.boxes_overlap(p['bbox'], v['bbox']):
                    key = self.get_condition_key("person_vehicle", p, v)
                    current.add(key)
        return current

    def _check_conditions_wirebond(self, detections_by_class):
        """Return a set of condition keys for wirebond area:
           - vacuume in/overlapping normal
           - vacuume in/overlapping suspected
           - spool in/overlapping grove
        """
        current = set()

        # vacuume + normal
        if 'vacuume' in detections_by_class and 'normal' in detections_by_class:
            for vdet in detections_by_class['vacuume']:
                for ndet in detections_by_class['normal']:
                    if self.is_inside_or_overlapping(vdet['bbox'], ndet['bbox']):
                        key = self.get_condition_key("vacuume_normal", vdet, ndet)
                        current.add(key)

        # vacuume + suspected
        if 'vacuume' in detections_by_class and 'suspected' in detections_by_class:
            for vdet in detections_by_class['vacuume']:
                for sdet in detections_by_class['suspected']:
                    if self.is_inside_or_overlapping(vdet['bbox'], sdet['bbox']):
                        key = self.get_condition_key("vacuume_suspected", vdet, sdet)
                        current.add(key)

        # spool + grove
        if 'spool' in detections_by_class and 'grove' in detections_by_class:
            for sdet in detections_by_class['spool']:
                for gdet in detections_by_class['grove']:
                    if self.is_inside_or_overlapping(sdet['bbox'], gdet['bbox']):
                        key = self.get_condition_key("spool_grove", sdet, gdet)
                        current.add(key)

        return current

    def _check_conditions_store(self, detections_by_class):
        """Return a set of condition keys for store area (any person detected)."""
        current = set()
        person_class = 'person'

        if person_class not in detections_by_class:
            return current

        for p in detections_by_class[person_class]:
            # Create a key based on the person's bounding box to track the same person across frames
            key = self.get_condition_key("person_detected", p)
            current.add(key)

        return current

    # ----------------------------------------------------------------------
    def process_detections(self, result, original_frame, frame_number,
                           video_fps, video_path, frame_size, cap):
        """Main detection processing – decides whether to save an image + clip."""
        if result.boxes is None:
            return None

        validation_frames = int(self.validation_time * video_fps)

        # Group detections by class, applying confidence threshold
        detections_by_class = {}
        for xyxy, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            if float(conf) < self.confidence_threshold:
                continue
            class_name = result.names[int(cls)]
            detections_by_class.setdefault(class_name, []).append({
                'bbox': xyxy,
                'confidence': float(conf)
            })

        # Call the appropriate condition checker
        if self.area == "scrap":
            current_conditions = self._check_conditions_scrap(detections_by_class)
        elif self.area == "wirebond":
            current_conditions = self._check_conditions_wirebond(detections_by_class)
        else:  # store
            current_conditions = self._check_conditions_store(detections_by_class)

        # Log new conditions
        for key in current_conditions:
            if key not in self.condition_frames:
                self.condition_frames[key] = frame_number
                self.logger.debug(f"Frame {frame_number}: New condition: {key}")

        # Remove expired conditions
        expired = [k for k in self.condition_frames if k not in current_conditions]
        for k in expired:
            self.logger.debug(f"Frame {frame_number}: Condition expired: {k}")
            self.validated_conditions.discard(k)
            del self.condition_frames[k]

        # Check for newly validated conditions
        should_save = False
        validated_key = None
        for key in current_conditions:
            if key in self.condition_frames:
                elapsed = frame_number - self.condition_frames[key]
                if elapsed >= validation_frames and key not in self.validated_conditions:
                    self.validated_conditions.add(key)
                    should_save = True
                    validated_key = key

        if should_save and validated_key:
            # Save original and annotated frames
            base = self.generate_filename("O")
            cv2.imwrite(os.path.join(self.save_dir, f"{base}.jpg"), original_frame)

            result_frame = result.plot(line_width=self.line_thickness)
            result_name = f"{self.generate_filename('R')}.jpg"
            cv2.imwrite(os.path.join(self.save_dir, result_name), result_frame)

            self.logger.info(f"Frame {frame_number}: saved {base}.jpg, {result_name}")

            # Save video clip centred on the detection
            detect_frame = self.condition_frames[validated_key]
            saved_frames = self.save_video_clip(video_path, detect_frame, base,
                                                video_fps, frame_size)

            # Compute restart frame (skip the clip we just saved)
            rewind_frames = int(self.clip_before_duration * video_fps)
            restart_frame = max(0, detect_frame - rewind_frames) + saved_frames
            self.logger.info(f"Clip centred at frame {detect_frame}, restarting at {restart_frame}")
            return restart_frame

        return None

    def process_video_file(self, video_path):
        """Process one video file completely."""
        video_name = os.path.basename(video_path)
        self.logger.info(f"Processing video: {video_name}")

        # Reset per‑video state
        self.condition_frames.clear()
        self.validated_conditions.clear()

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        if not cap.isOpened():
            self.logger.error(f"Failed to open {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        validation_frames = int(self.validation_time * video_fps)

        self.logger.info(f"Total frames: {total_frames}, FPS: {video_fps:.2f}, "
                         f"validation frames: {validation_frames}")

        start_time = time.time()
        frame_count = 0

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

                # YOLO inference
                results = self.model(frame, verbose=False)
                result = results[0]

                # Process detections – may return a restart frame
                restart = self.process_detections(result, frame, current_frame,
                                                  video_fps, video_path, frame_size, cap)
                if restart is not None and restart > current_frame:
                    self.logger.info(f"Jumping to frame {restart}")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, restart)
                    self.condition_frames.clear()
                    self.validated_conditions.clear()
                    continue

                # Display overlay (optional – can be disabled for headless operation)
                elapsed = time.time() - start_time
                proc_fps = frame_count / elapsed if elapsed > 0 else 0

                disp = result.plot(line_width=self.line_thickness)
                cv2.putText(disp, f"Proc FPS: {proc_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)
                cv2.putText(disp, f"Video: {video_name}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)
                cv2.putText(disp, f"Frame: {current_frame}/{total_frames}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)
                cv2.putText(disp, f"Validation frames: {validation_frames}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)
                cv2.imshow('Combined YOLO Inference', disp)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord(' '):
                    while True:
                        k2 = cv2.waitKey(0) & 0xFF
                        if k2 == ord(' '):
                            break
                        elif k2 == ord('q'):
                            self.running = False
                            break
                    if not self.running:
                        break

        except Exception as e:
            self.logger.error(f"Error in {video_name}: {e}")
        finally:
            cap.release()
            proc_time = time.time() - start_time
            self.logger.info(f"Finished {video_name}: {frame_count} frames in {proc_time:.2f}s "
                             f"({frame_count/proc_time:.2f} FPS)")

    def run(self):
        """Process all videos in the folder sequentially."""
        self.logger.info(f"Starting processing for area '{self.area}'")
        os.makedirs(self.save_dir, exist_ok=True)

        while self.running and self.current_video_index < len(self.video_files):
            video = self.video_files[self.current_video_index]
            self.process_video_file(video)
            if not self.running:
                break
            self.current_video_index += 1
            if self.current_video_index < len(self.video_files):
                self.logger.info(f"Moving to next video ({self.current_video_index+1}/{len(self.video_files)})")

        cv2.destroyAllWindows()
        self.logger.info("All videos processed.")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined security video analysis (scrap / wirebond / store)")
    parser.add_argument("area", choices=["scrap", "wirebond", "store"],
                        help="Area to analyse: 'scrap', 'wirebond', or 'store'")
    parser.add_argument("model_path", help="Path to YOLO model weights (e.g. yolov8n.pt or best.pt)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    parser.add_argument("--validation_time", type=float, default=2.0,
                        help="Seconds a condition must persist before saving (default: 2.0)")
    parser.add_argument("--line_thickness", type=int, default=1,
                        help="Thickness of bounding boxes (default: 1)")

    args = parser.parse_args()

    # Select video folder based on area
    if args.area == "scrap":
        video_folder = SCRAP_VIDEO_FOLDER
    elif args.area == "wirebond":
        video_folder = WIREBOND_VIDEO_FOLDER
    else:  # store
        video_folder = STORE_VIDEO_FOLDER

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Check paths
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found: {args.model_path}")
        exit(1)
    if not os.path.exists(video_folder):
        print(f"ERROR: Video folder not found: {video_folder}")
        exit(1)

    # Create detector instance
    detector = CombinedVideoYOLOInference(
        area=args.area,
        model_path=args.model_path,
        video_folder=video_folder,
        confidence_threshold=args.conf,
        line_thickness=args.line_thickness,
        validation_time=args.validation_time
    )

    try:
        detector.run()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cv2.destroyAllWindows()