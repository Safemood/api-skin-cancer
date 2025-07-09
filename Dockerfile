FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libexpat1 \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    ffmpeg \
    libfontconfig1 \
    libgl1-mesa-glx \
    libfreetype6 \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

EXPOSE 80

CMD ["gunicorn", "--bind", "0.0.0.0:80", "app:app"]
