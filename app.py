from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import cv2, numpy as np, io, torch

app = FastAPI()

# Chỉ load 1 model tốt nhất (cho nhanh) - worker nào cũng giống nhau
MODEL_WEIGHT = "yolov8n.pt"  # Hoặc yolov8s.pt, yolov8m.pt tùy accuracy
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Loading {MODEL_WEIGHT} on {device}")
model = YOLO(MODEL_WEIGHT)
model.to(device)
print("✅ Model ready!")

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    """
    Inference endpoint - mỗi worker xử lý độc lập
    """
    try:
        # Read image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Run inference
        result = model(img_cv, verbose=False)[0]
        
        # Parse results
        boxes = result.boxes.data.cpu().numpy().tolist()
        names = [result.names[int(cls)] for cls in result.boxes.cls.cpu().numpy()]
        
        return {
            "success": True,
            "model": MODEL_WEIGHT,
            "device": str(device),
            "boxes": boxes,
            "names": names,
            "inference_ms": round(result.speed['inference'], 2),
            "count": len(boxes)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/health")
async def health():
    """Health check cho RunPod"""
    return {
        "status": "healthy",
        "model": MODEL_WEIGHT,
        "device": str(device)
    }

@app.get("/")
async def root():
    return {"message": "YOLO inference ready!", "endpoint": "/infer"}
