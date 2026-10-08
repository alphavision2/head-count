import os, sys
deps = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deps")
if os.path.isdir(deps) and deps not in sys.path:
    sys.path.insert(0, deps)

import cascade_utils as cu
import face_recognition_module as frm
import numpy as np
import cv2

print("import main module test...")
import main  # noqa: F401
print("  import main OK")

print("cv2.face LBPH available:", frm.has_face_recognizer())

blank = np.zeros((400, 600, 3), dtype=np.uint8)
blank[:] = (200, 220, 255)
boxes, matched, present = frm.recognize_faces_from_frame(blank)
print("blank frame detect -> boxes:", len(boxes))

cap = cu.open_camera(0, width=160, height=120)
print("webcam opened:", cap is not None)
release_ok = True
if cap:
    try:
        cap.release()
    except Exception as e:
        print("release warn:", e)
        release_ok = False
print("cap release:", release_ok)
