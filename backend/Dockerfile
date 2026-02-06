# Backend Dockerfile for Render (or any container platform)
FROM python:3.11-slim

WORKDIR /app

# Install system deps for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY surgery_phase_detection/ ./surgery_phase_detection/
COPY backend/app.py ./app.py
COPY configs/ ./configs/

# Port
ENV PORT=10000
EXPOSE 10000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
