# backend/app.py
import os
import io
import time
import base64
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from ultralytics import YOLO
from threading import Lock

# --- PyTorch safe-globals allowlisting (fixes PyTorch 2.6+ unpickling errors) ---
import torch
from torch.serialization import add_safe_globals

try:
    from ultralytics.nn.tasks import DetectionModel
    add_safe_globals([DetectionModel])
except Exception:
    pass

add_safe_globals([torch.nn.modules.container.Sequential, torch.nn.modules.module.Module])

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("farm-detector")

# ---------- APP ----------
app = FastAPI(title="Farm Animal Detector API", version="1.0")

# NOTE: in production restrict origins to your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- CONFIG ----------
HERE = os.path.dirname(__file__)
DEFAULT_WEIGHT = os.path.join(HERE, "yolov8n.pt")   # fallback, ultralytics will download if needed
LOCAL_WEIGHT = os.path.join(HERE, "best.pt")       # put your trained farm-only model here if available

MODEL_PATH = LOCAL_WEIGHT if os.path.exists(LOCAL_WEIGHT) else DEFAULT_WEIGHT
logger.info("[startup] Loading model from: %s", MODEL_PATH)

# ---------- Allowed classes (farm animals + human)
# Edit this list to match the labels you want to allow in API responses.
ALLOWED_CLASSES = {
    "person", "cow", "cattle", "sheep", "goat", "pig", "horse", "chicken", "chick", "dog", "cat"
}

# alias map to normalize different label names -> canonical
CLASS_ALIAS = {
    "cattle": "cow",
    "cows": "cow",
    "sheeps": "sheep",
    "chickens": "chicken",
    "pigs": "pig",
    "goats": "goat",
    "horses": "horse",
    "chicks": "chick",
}

def normalize_label(name: str) -> str:
    if not name:
        return name
    n = name.strip().lower()
    return CLASS_ALIAS.get(n, n)

# ---------- Load model (raise helpful error if failing) ----------
try:
    model = YOLO(MODEL_PATH)
    logger.info("[startup] Model loaded. Names: %s", model.names)
except Exception as e:
    logger.exception("[startup] Error loading YOLO model:")
    raise

# Build allowed class ID set from model.names to avoid string mismatches
MODEL_NAMES = {int(k): v for k, v in getattr(model, "names", {}).items()} if getattr(model, "names", None) else {}
ALLOWED_CLASS_IDS = set()
missing_allowed = set()

for allowed in ALLOWED_CLASSES:
    # find any model class that normalizes to the allowed class
    found_any = False
    for cid, cname in MODEL_NAMES.items():
        if normalize_label(cname) == normalize_label(allowed):
            ALLOWED_CLASS_IDS.add(cid)
            found_any = True
    if not found_any:
        missing_allowed.add(allowed)

if missing_allowed:
    logger.warning("Some allowed classes were not found in model.names: %s. Use /debug-raw to inspect model classes.", missing_allowed)

logger.info("Allowed class ids: %s", sorted(list(ALLOWED_CLASS_IDS)))

# ---------- Tracking utils (IoU-based simple tracker) ----------
TRACK_IOU_THRESHOLD = 0.3
TRACK_MAX_AGE = 2.5  # seconds

tracker_lock = Lock()
tracks = {}  # id -> {'bbox':..., 'label':..., 'confidence':..., 'last_seen':ts}
next_track_id = 1

def now():
    return time.time()

def iou_box(a, b):
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)

def update_tracks(detections: List[Dict[str, Any]]):
    global next_track_id, tracks
    cur = now()
    results = []

    with tracker_lock:
        unmatched_tracks = list(tracks.items())  # (id, trackdict)
        new_tracks = {}

        for det in detections:
            bbox = det['bbox']
            label = det['label']
            conf = det['confidence']

            best_id = None
            best_iou = 0.0
            best_idx = -1

            for idx, (tid, t) in enumerate(unmatched_tracks):
                if t['label'] != label:
                    continue
                i = iou_box(bbox, t['bbox'])
                if i > best_iou:
                    best_iou = i
                    best_id = tid
                    best_idx = idx

            if best_id is not None and best_iou >= TRACK_IOU_THRESHOLD:
                t = unmatched_tracks.pop(best_idx)[1]
                t['bbox'] = bbox
                t['confidence'] = conf
                t['last_seen'] = cur
                new_tracks[best_id] = t
            else:
                tid = next_track_id
                next_track_id += 1
                new_tracks[tid] = {
                    'bbox': bbox,
                    'label': label,
                    'confidence': conf,
                    'last_seen': cur
                }

        # keep recent unmatched tracks
        for tid, t in unmatched_tracks:
            if cur - t['last_seen'] <= TRACK_MAX_AGE:
                new_tracks[tid] = t

        tracks = new_tracks

        for tid, t in tracks.items():
            results.append({
                'id': tid,
                'label': t['label'],
                'confidence': float(t.get('confidence', 0)),
                'bbox': [float(x) for x in t['bbox']]
            })

    return results

# ---------- Image utilities ----------
def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def image_bytes_to_cv2(b: bytes):
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

# ---------- Filtering & parsing ----------
# Tuning knobs
MIN_CONF = 0.25                  # Minimum confidence to accept detection (tune up if many false positives)
MIN_BOX_AREA_PIXELS = 500       # ignore tiny boxes (pixels); tune to your input resolution
MAX_UPLOAD_BYTES = 1024 * 1024 * 4  # 4 MB default limit (tune as needed)

def parse_ultralytics_result_filtered_by_id(r) -> List[Dict[str,Any]]:
    """
    Return detections filtered by:
      - class id in ALLOWED_CLASS_IDS
      - confidence >= MIN_CONF
      - box area >= MIN_BOX_AREA_PIXELS
    """
    dets = []
    if getattr(r, 'boxes', None) is None:
        return dets

    # vectorized path
    try:
        xyxy = r.boxes.xyxy.cpu().numpy()  # Nx4
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for box, c, cls in zip(xyxy, confs, clss):
            if float(c) < MIN_CONF:
                logger.debug("Filtered by conf: id=%s conf=%.3f", int(cls), float(c))
                continue
            if int(cls) not in ALLOWED_CLASS_IDS:
                logger.debug("Filtered by id: id=%s name=%s", int(cls), model.names.get(int(cls), str(int(cls))))
                continue
            # box area filter
            w = float(box[2]) - float(box[0])
            h = float(box[3]) - float(box[1])
            if (w * h) < MIN_BOX_AREA_PIXELS:
                logger.debug("Filtered by area: area=%.1f (w=%.1f h=%.1f)", (w*h), w, h)
                continue
            raw_label = model.names.get(int(cls), str(int(cls)))
            label = normalize_label(raw_label)
            dets.append({
                'label': label,
                'confidence': float(c),
                'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            })
    except Exception:
        # fallback: iterate r.boxes
        for box in r.boxes:
            xyxy = box.xyxy.cpu().numpy().tolist()[0]
            conf = float(box.conf.cpu().numpy().tolist()[0])
            cls = int(box.cls.cpu().numpy().tolist()[0])
            if conf < MIN_CONF:
                continue
            if cls not in ALLOWED_CLASS_IDS:
                continue
            w = float(xyxy[2]) - float(xyxy[0])
            h = float(xyxy[3]) - float(xyxy[1])
            if (w * h) < MIN_BOX_AREA_PIXELS:
                continue
            raw_label = model.names.get(cls, str(cls))
            label = normalize_label(raw_label)
            dets.append({
                'label': label,
                'confidence': conf,
                'bbox': [xyxy[0], xyxy[1], xyxy[2], xyxy[3]]
            })
    return dets

# ---------- Endpoints ----------
@app.get("/status")
async def status():
    names = [normalize_label(n) for n in model.names.values()] if getattr(model, "names", None) else []
    return {"ok": True, "model_path": MODEL_PATH, "model_names": names, "allowed_classes": sorted(list(ALLOWED_CLASSES)), "allowed_class_ids": sorted(list(ALLOWED_CLASS_IDS))}

@app.get("/classes")
async def classes():
    """Return the allowed class names (useful for frontend legend)."""
    return {"allowed": sorted(list(ALLOWED_CLASSES)), "allowed_ids": sorted(list(ALLOWED_CLASS_IDS))}

@app.post("/debug-raw")
async def debug_raw(file: UploadFile = File(...), conf: float = Query(0.15)):
    """
    Debug endpoint: returns raw detections (class ids + names + conf + bbox)
    Use this to inspect what the model predicts for a given image.
    """
    b = await file.read()
    if len(b) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "payload too large"}, status_code=413)
    img_cv = image_bytes_to_cv2(b)
    results = model.predict(source=img_cv, conf=conf, verbose=False)
    r = results[0]
    raw = []
    try:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for box, c, cls in zip(xyxy, confs, clss):
            raw.append({
                "class_id": int(cls),
                "class_name": model.names.get(int(cls), str(int(cls))),
                "confidence": float(c),
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            })
    except Exception:
        for box in r.boxes:
            xyxy = box.xyxy.cpu().numpy().tolist()[0]
            conf = float(box.conf.cpu().numpy().tolist()[0])
            cls = int(box.cls.cpu().numpy().tolist()[0])
            raw.append({
                "class_id": cls,
                "class_name": model.names.get(cls, str(cls)),
                "confidence": conf,
                "bbox": xyxy
            })
    return {"raw": raw, "model_names": model.names}

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = Query(0.15)):
    b = await file.read()
    if len(b) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "payload too large"}, status_code=413)
    # enforce minimum confidence server-side
    conf = max(conf, MIN_CONF)
    img = image_bytes_to_cv2(b)
    results = model.predict(source=img, conf=conf, verbose=False)
    r = results[0]
    dets = parse_ultralytics_result_filtered_by_id(r)
    return JSONResponse({"predictions": dets})

@app.post("/annotated")
async def annotated(file: UploadFile = File(...), conf: float = Query(0.15)):
    b = await file.read()
    if len(b) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "payload too large"}, status_code=413)

    conf = max(conf, MIN_CONF)
    img_pil = pil_from_bytes(b)
    img_cv = image_bytes_to_cv2(b)
    results = model.predict(source=img_cv, conf=conf, verbose=False)
    r = results[0]
    dets = parse_ultralytics_result_filtered_by_id(r)

    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for d in dets:
        x1,y1,x2,y2 = d['bbox']
        label = f"{d['label']} {d['confidence']:.2f}"
        draw.rectangle([x1,y1,x2,y2], outline="red", width=2)
        draw.text((x1, y1 - 10), label, fill="white", font=font)

    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return {"image_base64": b64, "predictions": dets}

@app.post("/frame-detect")
async def frame_detect(file: UploadFile = File(...), conf: float = Query(0.15)):
    b = await file.read()
    if len(b) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "payload too large"}, status_code=413)

    conf = max(conf, MIN_CONF)
    img_cv = image_bytes_to_cv2(b)
    results = model.predict(source=img_cv, conf=conf, verbose=False)
    r = results[0]
    dets = parse_ultralytics_result_filtered_by_id(r)

    tracks_out = update_tracks(dets)

    return {"predictions": dets, "tracks": tracks_out}

# Optional: health endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}

# ---------- Run (dev) ----------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
