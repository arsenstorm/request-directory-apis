FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y wget && \
    wget https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.onnx && \
    apt-get remove -y wget && apt-get autoremove -y && apt-get clean

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

# Expose the port the app runs on
EXPOSE 7001

# Set environment variables
ENV NUDENET_USE_640M=false
ENV NUDENET_DEBUG=false
ENV NUDENET_PORT=7001

CMD ["python", "src/main.py"]
