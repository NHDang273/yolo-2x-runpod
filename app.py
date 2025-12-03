from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import cv2, numpy as np, io, torch
import os
import uvicorn

app = FastAPI()

# Load model 1 lần
MODEL_WEIGHT = "yolov8n.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Loading {MODEL_WEIGHT} on {device}")
model = YOLO(MODEL_WEIGHT)
model.to(device)
print("✅ Model ready!")

@app.get("/ping")
async def ping():
    """Health check endpoint - REQUIRED by RunPod Load Balancing"""
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "YOLO inference ready!", "endpoint": "/infer"}

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    """Inference endpoint"""
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

if __name__ == "__main__":
    # Lấy port từ environment variable (RunPod Load Balancing requirement)
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        workers=1
    )
