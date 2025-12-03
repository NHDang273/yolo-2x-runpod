FROM runpod/pytorch:2.4.0-py3.10-cuda12.4.1-devel-ubuntu22.04

# Install deps từ requirements.txt (RunPod tự copy từ repo)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app.py và models (RunPod tự copy)
COPY app.py .
COPY *.pt .  # Copy tất cả file .pt (yolov8n.pt, yolov8s.pt)

# Expose port
EXPOSE 8000

# Start command: Chạy uvicorn cho FastAPI (app.py)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
