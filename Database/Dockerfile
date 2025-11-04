FROM python:3.10-slim

# Install library MongoDB
RUN pip install pymongo

# Copy script Python ke dalam container
COPY face_log_to_mongo.py /app/face_log_to_mongo.py

WORKDIR /app

# Jalankan script
CMD ["python3", "face_log_to_mongo.py"]
