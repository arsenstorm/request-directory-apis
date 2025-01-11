FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libomp-dev \
    libopenblas-dev \
    wget && \
    wget https://github.com/<insert> && \
    apt-get remove -y wget && apt-get autoremove -y && apt-get clean

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

# Expose the port the app runs on
EXPOSE 7004

# Set environment variables
ENV IMAGETALKER_DEBUG=false
ENV IMAGETALKER_PORT=7004

# If you are using Replicate to run Image Talker
# then you can set the variables here:
ENV IMAGETALKER_RUN_LOCAL=false
ENV IMAGETALKER_REPLICATE_MODEL=imagetalker
ENV IMAGETALKER_REPLICATE_API_KEY=sk...

CMD ["python", "src/main.py"]
