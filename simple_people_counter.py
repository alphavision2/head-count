import os
import sys

_DEPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deps")
if os.path.isdir(_DEPS_DIR) and _DEPS_DIR not in sys.path:
    sys.path.insert(0, _DEPS_DIR)

import cv2
import numpy as np

from cascade_utils import open_camera, build_haar_cascade


def _mediapipe_detect(frame_rgb):
    try:
        import mediapipe as mp
    except Exception:
        return None
    h, w = frame_rgb.shape[:2]
    try:
        detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.55
        )
    except Exception:
        return None
    try:
        results = detector.process(frame_rgb)
    except Exception:
        return None
    boxes = []
    if results and getattr(results, "detections", None):
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x = int(max(0, bb.xmin * w))
            y = int(max(0, bb.ymin * h))
            bw = int(min(w - x, bb.width * w))
            bh = int(min(h - y, bb.height * h))
            if bw < 20 or bh < 20:
                continue
            boxes.append((x, y, bw, bh))
    try:
        detector.close()
    except Exception:
        pass
    return boxes


def _haar_detect(gray, cascade):
    h, w = gray.shape[:2]
    minsz = (max(50, int(h * 0.08)), max(50, int(h * 0.08)))
    try:
        rects = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                         minSize=minsz, flags=cv2.CASCADE_SCALE_IMAGE)
    except Exception:
        return []
    if rects is None or len(rects) == 0:
        return []
    return [tuple(map(int, r)) for r in rects]


def count_people_webcam(camera_index=0):
    haar_cascade, haar_path = build_haar_cascade()
    if haar_cascade is None or haar_cascade.empty():
        haar_cascade = None

    cap = open_camera(camera_index, width=960, height=720)
    if cap is None:
        print("ERROR: Could not open any webcam.\n"
              " - Try a different USB port.\n"
              " - If you have multiple cameras, run: python simple_people_counter.py 1")
        return

    print("\n=== Live People Counter Started ===")
    print(f"Webcam opened. Haar cascade: {'OK' if haar_cascade else 'N/A (using MediaPipe)'}\n"
          "Press q on the video window to quit, s to save snapshot.\n")

    smoothed = 0
    alpha = 0.4
    last_counts = []
    use_mp_every = 2
    frame_i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost webcam signal.")
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = None
        if frame_i % use_mp_every == 0:
            faces = _mediapipe_detect(rgb)
        if not faces and haar_cascade is not None:
            faces = _haar_detect(gray, haar_cascade)
        if not faces:
            faces = []

        frame_i += 1

        last_counts.append(len(faces))
        if len(last_counts) > 7:
            last_counts.pop(0)
        avg = int(round(sum(last_counts) / len(last_counts))) if last_counts else 0
        smoothed = int(round((1 - alpha) * smoothed + alpha * avg))

        for idx, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 90), 2)
            cv2.putText(frame, f"Person {idx + 1}", (x, max(y - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 90), 2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.putText(frame, "People in front of camera:", (25, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 220, 255), 2)
        cv2.putText(frame, f"  COUNT = {smoothed}", (25, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 140), 3)

        cv2.imshow("Head Count - Webcam Live (q=quit, s=snapshot)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(f"Final count: {smoothed} people detected.")
            break
        if key == ord('s'):
            fname = f"headcount_snapshot_{cv2.getTickCount()}.jpg"
            cv2.imwrite(fname, frame)
            print(f"Snapshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cam = 0
    if len(sys.argv) > 1:
        try:
            cam = int(sys.argv[1])
        except ValueError:
            pass
    count_people_webcam(camera_index=cam)
