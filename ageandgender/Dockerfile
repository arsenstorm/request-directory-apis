FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libomp-dev \
    libopenblas-dev && \
    apt-get autoremove -y && apt-get clean

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

# Expose the port the app runs on
EXPOSE 7003

# Set environment variables
ENV AGEANDGENDER_DEBUG=false
ENV AGEANDGENDER_PORT=7003

CMD ["python", "src/main.py"]
