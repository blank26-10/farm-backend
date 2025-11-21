# backend/app.py
import os
import io
import time
import base64
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
# ultralytics internal class that may be stored in .pt files
try:
    from ultralytics.nn.tasks import DetectionModel
    add_safe_globals([DetectionModel])
except Exception:
    # if import fails, we'll still attempt to add common torch classes below
    pass

# allowlist common torch container used inside many checkpoints
add_safe_globals([torch.nn.modules.container.Sequential, torch.nn.modules.module.Module])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- CONFIG ----------
HERE = os.path.dirname(__file__)
DEFAULT_WEIGHT = os.path.join(HERE, "yolov8n.pt")  # fallback weight (will be downloaded by ultralytics if absent)
LOCAL_WEIGHT = os.path.join(HERE, "best.pt")  # if you trained and placed a best.pt here

MODEL_PATH = LOCAL_WEIGHT if os.path.exists(LOCAL_WEIGHT) else DEFAULT_WEIGHT
print(f"Loading model from: {MODEL_PATH}")

# load model (wrapped to surface clear error if something else needs allowlisting)
try:
    model = YOLO(MODEL_PATH)  # will use CPU or GPU if available
    print("Model loaded. Names:", model.names)
except Exception as e:
    # Re-raise with a helpful message for logs
    print("Error loading YOLO model. If you see a torch UnpicklingError complaining about 'Unsupported global',")
    print("you need to add that class to torch.serialization.add_safe_globals([...]) before creating the YOLO object.")
    raise

# ---------- SIMPLE SERVER-SIDE TRACKER (IoU-based) ----------
TRACK_IOU_THRESHOLD = 0.3
TRACK_MAX_AGE = 2.5  # seconds

tracker_lock = Lock()
tracks = {}  # id -> {'bbox':[x1,y1,x2,y2], 'label':str, 'confidence':float, 'last_seen':ts}
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

# ---------- DETECTION UTIL ----------
def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def image_bytes_to_cv2(b: bytes):
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def parse_ultralytics_result(r) -> List[Dict[str,Any]]:
    dets = []
    if getattr(r, 'boxes', None) is None:
        return dets
    # ultralytics r.boxes may provide tensors
    try:
        xyxy = r.boxes.xyxy.cpu().numpy()  # Nx4
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for box, c, cls in zip(xyxy, confs, clss):
            label = model.names.get(int(cls), str(int(cls)))
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
            label = model.names.get(cls, str(cls))
            dets.append({
                'label': label,
                'confidence': conf,
                'bbox': [xyxy[0], xyxy[1], xyxy[2], xyxy[3]]
            })
    return dets

# ---------- ENDPOINTS ----------
@app.get("/status")
async def status():
    return {"ok": True, "model": MODEL_PATH, "names": model.names}

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = Query(0.25)):
    b = await file.read()
    img = image_bytes_to_cv2(b)
    results = model.predict(source=img, conf=conf, verbose=False)
    r = results[0]
    dets = parse_ultralytics_result(r)
    return JSONResponse({"predictions": dets})

@app.post("/annotated")
async def annotated(file: UploadFile = File(...), conf: float = Query(0.25)):
    b = await file.read()
    img_pil = pil_from_bytes(b)
    img_cv = image_bytes_to_cv2(b)
    results = model.predict(source=img_cv, conf=conf, verbose=False)
    r = results[0]
    dets = parse_ultralytics_result(r)

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
async def frame_detect(file: UploadFile = File(...), conf: float = Query(0.25)):
    b = await file.read()
    img_cv = image_bytes_to_cv2(b)
    results = model.predict(source=img_cv, conf=conf, verbose=False)
    r = results[0]
    dets = parse_ultralytics_result(r)

    tracks_out = update_tracks(dets)

    return {"predictions": dets, "tracks": tracks_out}

# OPTIONAL: video endpoints could be added here (video-annotated, video-detect) if needed

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
