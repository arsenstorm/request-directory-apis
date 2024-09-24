# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables for the PyTorch build
ENV USE_NNPACK=0

# Copy everything
COPY . .

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libomp-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install -r requirements.txt

# Expose the port the app runs on
EXPOSE 7002

# Set environment variables
ENV FACELANDMARKS_DEBUG=false
ENV FACELANDMARKS_PORT=7002

# Run the application
CMD ["python", "src/main.py"]