import os
import sys
import urllib.request
import ssl


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADES_DIR = os.path.join(_THIS_DIR, "cascades")
HAAR_FILE = "haarcascade_frontalface_default.xml"
HAAR_URL = ("https://raw.githubusercontent.com/opencv/opencv/master/"
            "data/haarcascades/" + HAAR_FILE)


def ensure_cascades_dir():
    try:
        os.makedirs(CASCADES_DIR, exist_ok=True)
    except OSError:
        pass


def _check_path(path):
    return bool(path and os.path.isfile(path) and os.path.getsize(path) > 5000)


def _find_system_haar_xml():
    try:
        import cv2
        # Standard cv2.data location
        p = os.path.join(getattr(cv2.data, "haarcascades", ""), HAAR_FILE)
        if _check_path(p):
            return p
    except Exception:
        pass
    # Common site-packages / pip cache install locations
    base_prefixes = []
    try:
        base_prefixes.append(sys.prefix)
        base_prefixes.append(sys.base_prefix)
    except Exception:
        pass
    candidates = []
    for pr in base_prefixes:
        if not pr:
            continue
        candidates.append(os.path.join(pr, "Lib", "site-packages", "cv2", "data", HAAR_FILE))
        candidates.append(os.path.join(pr, "cv2", "data", HAAR_FILE))
        candidates.append(os.path.join(pr, "data", HAAR_FILE))
    for pr in os.environ.get("PATH", "").split(os.pathsep):
        if not pr:
            continue
        candidates.append(os.path.join(pr, "cv2", "data", HAAR_FILE))
        candidates.append(os.path.join(pr, "data", HAAR_FILE))
    for c in candidates:
        if _check_path(c):
            return c
    return None


def _download_haar_xml(destination):
    ensure_cascades_dir()
    try:
        ctx = ssl._create_unverified_context() if hasattr(ssl, "_create_unverified_context") else None
        req = urllib.request.Request(HAAR_URL, headers={"User-Agent": "HeadCount/1.0"})
        with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
            data = resp.read()
        if len(data) < 10000:
            return False
        with open(destination, "wb") as f:
            f.write(data)
        return _check_path(destination)
    except Exception:
        return False


def get_haar_cascade_path(force_refresh=False):
    local = os.path.join(CASCADES_DIR, HAAR_FILE)
    if not force_refresh and _check_path(local):
        return local
    sys_path = _find_system_haar_xml()
    if sys_path:
        try:
            ensure_cascades_dir()
            import shutil
            shutil.copy2(sys_path, local)
            if _check_path(local):
                return local
        except Exception:
            return sys_path
    if _download_haar_xml(local):
        return local
    return sys_path or local or None


def build_haar_cascade():
    path = get_haar_cascade_path()
    if not path or not os.path.isfile(path):
        return None, path
    import cv2
    c = cv2.CascadeClassifier(path)
    if c.empty():
        # Corrupt file: re-download once
        path2 = get_haar_cascade_path(force_refresh=True)
        if path2 and os.path.isfile(path2):
            c = cv2.CascadeClassifier(path2)
            if not c.empty():
                return c, path2
        return None, path
    return c, path


def open_camera(index=0, width=960, height=720):
    import cv2
    backends = []
    try:
        backends.append(cv2.CAP_DSHOW)
    except Exception:
        pass
    backends.append(0)
    indexes = [int(index)] if isinstance(index, int) or str(index).isdigit() else [0, 1]
    for idx in indexes:
        for backend in backends:
            try:
                if backend == 0:
                    cap = cv2.VideoCapture(idx)
                else:
                    cap = cv2.VideoCapture(idx, backend)
            except Exception:
                continue
            if cap is None:
                continue
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            except Exception:
                pass
            ok, _ = cap.read()
            if ok:
                return cap
            try:
                cap.release()
            except Exception:
                pass
    return None
