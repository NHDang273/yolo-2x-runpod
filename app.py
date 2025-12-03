from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import cv2, numpy as np, io, os, torch

app = FastAPI()

# Danh sách 2 model YOLO
MODELS = [
    "yolov8n.pt",   # GPU 0
    "yolov8s.pt"    # GPU 1
]

# Load models vào từng GPU
models = []
for i, weight in enumerate(MODELS):
    device = f"cuda:{i}"
    print(f"Loading {weight} → {device}")
    model = YOLO(weight)  # Tự tải nếu file chưa có, hoặc load từ local
    model.to(device)
    models.append(model)

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Load balance bằng hash
    gpu_id = hash(contents) % len(models)
    result = models[gpu_id](img_cv, verbose=False)[0]

    boxes = result.boxes.data.cpu().numpy().tolist()
    names = [result.names[int(cls)] for cls in result.boxes.cls.cpu().numpy()]

    return {
        "gpu_used": gpu_id,
        "model": MODELS[gpu_id],
        "boxes": boxes,
        "names": names,
        "inference_ms": round(result.speed['inference'], 2)
    }

@app.get("/")
async def root():
    return {"message": "2x YOLO ready! POST image to /infer"}