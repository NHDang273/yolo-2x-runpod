# ==== DÒNG NÀY LÀ QUAN TRỌNG NHẤT – TAG CHUẨN RUNPOD 2025 ====
FROM runpod/pytorch:2.4.1-py3.10-cuda12.4.1-ubuntu22.04

# Đặt thư mục làm việc
WORKDIR /app

# Copy và cài đặt dependencies trước để tận dụng cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code chính
COPY app.py .

# (Tùy chọn) Nếu bạn sau này muốn dùng custom .pt thì uncomment dòng dưới
# COPY *.pt ./

# Mở port cho FastAPI
EXPOSE 8000

# Khởi động server (RunPod sẽ dùng cái này)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
