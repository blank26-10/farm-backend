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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- CONFIG ----------
HERE = os.path.dirname(__file__)
DEFAULT_WEIGHT = os.path.join(HERE, "yolov8n.pt")
LOCAL_WEIGHT = os.path.join(HERE, "best.pt")

MODEL_PATH = LOCAL_WEIGHT if os.path.exists(LOCAL_WEIGHT) else DEFAULT_WEIGHT
print(f"Loading model from: {MODEL_PATH}")

model = YOLO(MODEL_PATH)
print("Model loaded. Names:", model.names)

# ---------- SIMPLE TRACKER ----------
TRACK_IOU_THRESHOLD = 0.3
TRACK_MAX_AGE = 2.5  

tracker_lock = Lock()
tracks = {}
next_track_id = 1

def now(): return time.time()

def iou_box(a, b):
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter = inter_w * inter_h
    if inter == 0: return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)

def update_tracks(detections):
    global next_track_id, tracks
    cur = now()
    results = []

    with tracker_lock:
        unmatched = list(tracks.items())
        new_tracks = {}

        for det in detections:
            bbox = det['bbox']
            label = det['label']
            conf = det['confidence']

            best_id = None
            best_iou = 0
            best_idx = -1

            for idx, (tid, t) in enumerate(unmatched):
                if t['label'] != label:
                    continue
                i = iou_box(bbox, t['bbox'])
                if i > best_iou:
                    best_iou = i
                    best_id = tid
                    best_idx = idx

            if best_id is not None and best_iou >= TRACK_IOU_THRESHOLD:
                item = unmatched.pop(best_idx)[1]
                item['bbox'] = bbox
                item['confidence'] = conf
                item['last_seen'] = cur
                new_tracks[best_id] = item
            else:
                tid = next_track_id
                next_track_id += 1
                new_tracks[tid] = {
                    'bbox': bbox,
                    'label': label,
                    'confidence': conf,
                    'last_seen': cur
                }

        for tid, t in unmatched:
            if cur - t['last_seen'] <= TRACK_MAX_AGE:
                new_tracks[tid] = t

        tracks = new_tracks

        for tid, t in tracks.items():
            results.append({
                'id': tid,
                'label': t['label'],
                'confidence': float(t['confidence']),
                'bbox': [float(x) for x in t['bbox']]
            })

    return results


# ---------- UTIL ----------
def pil_from_bytes(b): return Image.open(io.BytesIO(b)).convert("RGB")

def image_bytes_to_cv2(b):
    arr = np.frombuffer(b, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def parse_ultralytics_result(r):
    dets = []
    if r.boxes is None:
        return dets
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy.cpu().numpy().tolist()[0]
        conf = float(box.conf)
        cls = int(box.cls)
        label = model.names.get(cls, str(cls))
        dets.append({
            'label': label,
            'confidence': conf,
            'bbox': [x1, y1, x2, y2]
        })
    return dets


# ---------- ENDPOINTS ----------
@app.get("/status")
async def status():
    return {"ok": True, "model": MODEL_PATH, "names": model.names}

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = Query(0.15)):
    b = await file.read()
    img = image_bytes_to_cv2(b)
    results = model.predict(source=img, conf=conf, verbose=False)[0]

    dets = parse_ultralytics_result(results)
    return {"predictions": dets}


@app.post("/annotated")
async def annotated(file: UploadFile = File(...), conf: float = Query(0.15)):
    b = await file.read()
    img_pil = pil_from_bytes(b)
    img_cv = image_bytes_to_cv2(b)

    results = model.predict(source=img_cv, conf=conf, verbose=False)[0]
    dets = parse_ultralytics_result(results)

    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.load_default()

    for d in dets:
        x1, y1, x2, y2 = d['bbox']
        label = f"{d['label']} {d['confidence']:.1f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), label, fill="white", font=font)

    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return {"image_base64": b64, "predictions": dets}


# ---------- FIXED LIVE FRAME DETECTION ----------
@app.post("/frame-detect")
async def frame_detect(file: UploadFile = File(...), conf: float = Query(0.15)):
    b = await file.read()
    img_cv = image_bytes_to_cv2(b)

    # ---------------- FIX: Speed up real-time detection ----------------
    h, w = img_cv.shape[:2]
    scale = 0.5   # reduce frame size → faster YOLO
    img_small = cv2.resize(img_cv, (int(w * scale), int(h * scale)))
    # -------------------------------------------------------------------

    results = model.predict(source=img_small, conf=conf, verbose=False)[0]
    dets = parse_ultralytics_result(results)

    # Re-scale detections back to original image size
    for d in dets:
        d['bbox'][0] /= scale
        d['bbox'][1] /= scale
        d['bbox'][2] /= scale
        d['bbox'][3] /= scale

    tracks_out = update_tracks(dets)

    return {"predictions": dets, "tracks": tracks_out}


# ---------- MAIN ----------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
