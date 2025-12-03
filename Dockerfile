FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Expose cả 2 ports (PORT và PORT_HEALTH có thể khác nhau)
EXPOSE 8000

# Chạy với port từ env variable
CMD ["python", "app.py"]
