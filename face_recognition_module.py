import os
import json
import shutil
import pickle
import numpy as np
import cv2


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STUDENTS_FILE = os.path.join(DATA_DIR, "students.json")
FACES_FILE = os.path.join(DATA_DIR, "registered_faces.pkl")
IMAGES_DIR = os.path.join(DATA_DIR, "student_images")
RECOG_SIZE = (160, 160)

_haar_cascade = None
_mp_detector = None
_lbph_recognizer = None
_id_to_label = {}
_label_to_id = {}
_has_cv2_face = hasattr(cv2, "face") and hasattr(getattr(cv2, "face", None), "LBPHFaceRecognizer_create")


def has_face_recognizer():
    return _has_cv2_face


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)


def load_students():
    ensure_dirs()
    if os.path.exists(STUDENTS_FILE):
        with open(STUDENTS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_students(students):
    ensure_dirs()
    with open(STUDENTS_FILE, "w") as f:
        json.dump(students, f, indent=4)


def _get_haar_cascade():
    global _haar_cascade
    if _haar_cascade is None or getattr(_haar_cascade, "_hc_path_bad", False):
        from cascade_utils import build_haar_cascade
        cascade, path = build_haar_cascade()
        if cascade is None or cascade.empty():
            _haar_cascade = None
            return None
        _haar_cascade = cascade
    return _haar_cascade


def _get_mp_detector():
    global _mp_detector
    if _mp_detector is False:
        return None
    if _mp_detector is None:
        try:
            import mediapipe as mp  # noqa: F401
            _mp_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.55
            )
        except Exception:
            _mp_detector = False
    return _mp_detector


def _detect_faces_haar(rgb_image):
    h, w = rgb_image.shape[:2]
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    min_size = (max(40, int(h * 0.08)), max(40, int(h * 0.08)))
    cascade = _get_haar_cascade()
    if cascade is None:
        return []
    try:
        rects = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                         minSize=min_size, flags=cv2.CASCADE_SCALE_IMAGE)
    except Exception:
        return []
    boxes = []
    if rects is None:
        return boxes
    for (x, y, bw, bh) in rects:
        boxes.append((y, x + bw, y + bh, x))
    return boxes


def _detect_faces_mediapipe(rgb_image):
    h, w = rgb_image.shape[:2]
    detector = _get_mp_detector()
    if detector is None:
        return []
    try:
        results = detector.process(rgb_image)
    except Exception:
        return []
    boxes = []
    if results.detections:
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x = int(max(0, bb.xmin * w))
            y = int(max(0, bb.ymin * h))
            bw = int(min(w - x, bb.width * w))
            bh = int(min(h - y, bb.height * h))
            if bw < 20 or bh < 20:
                continue
            boxes.append((y, x + bw, y + bh, x))
    return boxes


def detect_all_faces(rgb_image):
    boxes = _detect_faces_mediapipe(rgb_image)
    if not boxes:
        boxes = _detect_faces_haar(rgb_image)
    return boxes


def _crop_face(rgb_image, box, pad=0.08):
    top, right, bottom, left = box
    h, w = rgb_image.shape[:2]
    bw = right - left
    bh = bottom - top
    pad_x = int(bw * pad)
    pad_y = int(bh * pad)
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(w, right + pad_x)
    bottom = min(h, bottom + pad_y)
    face = rgb_image[top:bottom, left:right]
    if face.size == 0:
        return None
    return face


def _preprocess_face(rgb_face_crop):
    gray = cv2.cvtColor(rgb_face_crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    resized = cv2.resize(gray, RECOG_SIZE, interpolation=cv2.INTER_CUBIC)
    return resized


def _load_faces_db():
    ensure_dirs()
    if os.path.exists(FACES_FILE):
        with open(FACES_FILE, "rb") as f:
            return pickle.load(f)
    return {"ids": [], "faces": []}


def _save_faces_db(db):
    ensure_dirs()
    with open(FACES_FILE, "wb") as f:
        pickle.dump(db, f)


def _build_recognizer():
    global _lbph_recognizer, _id_to_label, _label_to_id
    _lbph_recognizer = None
    _id_to_label = {}
    _label_to_id = {}
    if not _has_cv2_face:
        return None
    db = _load_faces_db()
    if len(db["faces"]) == 0:
        return None
    faces = []
    labels = []
    next_label = 0
    for sid, face_img in zip(db["ids"], db["faces"]):
        if sid not in _id_to_label:
            _id_to_label[sid] = next_label
            _label_to_id[next_label] = sid
            next_label += 1
        faces.append(face_img)
        labels.append(_id_to_label[sid])
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=95.0
    )
    recognizer.train(faces, np.array(labels, dtype=np.int32))
    _lbph_recognizer = recognizer
    return recognizer


def _predict_face(processed_face, strict_threshold=80.0):
    if not _has_cv2_face:
        return None, float("inf")
    if _lbph_recognizer is None:
        _build_recognizer()
    if _lbph_recognizer is None:
        return None, float("inf")
    try:
        label, confidence = _lbph_recognizer.predict(processed_face)
    except Exception:
        return None, float("inf")
    if confidence <= strict_threshold and label in _label_to_id:
        return _label_to_id[label], confidence
    return None, confidence


def register_student(student_id, name, image_path):
    students = load_students()
    if student_id in students:
        return False, f"Student ID {student_id} already exists."

    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return False, "Could not read the image file."
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes = detect_all_faces(rgb)
    if len(boxes) == 0:
        return False, "No face detected in the image. Try a clearer front-facing photo."
    if len(boxes) > 1:
        return False, "Multiple faces detected. Please upload an image with a single face."

    face_crop = _crop_face(rgb, boxes[0])
    if face_crop is None:
        return False, "Could not crop the detected face."
    processed = _preprocess_face(face_crop)

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".bmp"):
        ext = ".jpg"
    saved_image_path = os.path.join(IMAGES_DIR, f"{student_id}{ext}")
    shutil.copy2(image_path, saved_image_path)

    students[student_id] = {
        "id": student_id,
        "name": name,
        "image_path": saved_image_path
    }
    save_students(students)

    db = _load_faces_db()
    db["ids"].append(student_id)
    db["faces"].append(processed)
    _save_faces_db(db)

    if _has_cv2_face:
        _build_recognizer()
        return True, f"Student {name} (ID: {student_id}) registered successfully.\nFace recognition is ENABLED."
    else:
        return True, (
            f"Student {name} (ID: {student_id}) registered successfully.\n"
            "INFO: 'opencv-contrib-python' is not installed yet, so face-by-name\n"
            "recognition is disabled. Total-face counting still works!\n"
            "To enable recognition, run:\n"
            "  pip install --target=deps opencv-contrib-python"
        )


def delete_student(student_id):
    students = load_students()
    if student_id not in students:
        return False, f"Student ID {student_id} not found."

    image_path = students[student_id]["image_path"]
    if os.path.exists(image_path):
        try:
            os.remove(image_path)
        except OSError:
            pass

    del students[student_id]
    save_students(students)

    db = _load_faces_db()
    new_ids, new_faces = [], []
    for sid, face in zip(db["ids"], db["faces"]):
        if sid != student_id:
            new_ids.append(sid)
            new_faces.append(face)
    db["ids"], db["faces"] = new_ids, new_faces
    _save_faces_db(db)

    if _has_cv2_face:
        _build_recognizer()
    return True, f"Student ID {student_id} deleted successfully."


def recognize_faces_in_image(image_path, threshold=80.0):
    students = load_students()
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, "Could not read the image file."
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes = detect_all_faces(rgb)
    present_ids = []
    unknown = 0

    if _has_cv2_face:
        if _lbph_recognizer is None:
            _build_recognizer()
        if len(students) == 0:
            result = {
                "present": [], "present_count": 0,
                "total_registered": 0,
                "total_faces_detected": len(boxes), "unknown_faces": len(boxes)
            }
            return result, None
        for box in boxes:
            crop = _crop_face(rgb, box)
            if crop is None:
                unknown += 1
                continue
            proc = _preprocess_face(crop)
            sid, _ = _predict_face(proc, strict_threshold=threshold)
            if sid is not None and sid in students and sid not in present_ids:
                present_ids.append(sid)
            elif sid is None:
                unknown += 1
    else:
        unknown = len(boxes)

    present_details = [students[sid] for sid in present_ids if sid in students]
    result = {
        "present": present_details,
        "present_count": len(present_details),
        "total_registered": len(students),
        "total_faces_detected": len(boxes),
        "unknown_faces": unknown
    }
    return result, None


def recognize_faces_from_frame(rgb_frame, threshold=80.0):
    students = load_students()
    boxes = detect_all_faces(rgb_frame)
    matched_ids = []
    present_ids = []

    if _has_cv2_face and len(students) > 0:
        if _lbph_recognizer is None:
            _build_recognizer()
        for box in boxes:
            crop = _crop_face(rgb_frame, box)
            if crop is None:
                matched_ids.append(None)
                continue
            proc = _preprocess_face(crop)
            sid, _ = _predict_face(proc, strict_threshold=threshold)
            matched_ids.append(sid)
            if sid is not None and sid in students and sid not in present_ids:
                present_ids.append(sid)
    else:
        matched_ids = [None] * len(boxes)

    return boxes, matched_ids, present_ids
