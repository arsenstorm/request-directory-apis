# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Clone the PyTorch repository
RUN git clone --recursive https://github.com/pytorch/pytorch && \
    cd pytorch && \
    git submodule sync && \
    git submodule update --init --recursive

# Set environment variables for the PyTorch build
ENV USE_NNPACK=0

# Install dependencies for building PyTorch
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libomp-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch from source
WORKDIR /pytorch
RUN python setup.py install

# Return to the /app directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 7002

# Set environment variables
ENV FACELANDMARKS_DEBUG=false
ENV FACELANDMARKS_PORT=7002

# Run the application
CMD ["python", "src/main.py"]