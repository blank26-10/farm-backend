# 1. Use official Python
FROM python:3.10

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies (needed for OpenCV & YOLO)
RUN apt-get update && apt-get install -y \
    libgl1\
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements file
COPY requirements.txt .

# 5. Install pip packages
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy all backend files to container
COPY . .

# 7. Expose port (Render auto-maps this)
EXPOSE 8000

# 8. Run FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
