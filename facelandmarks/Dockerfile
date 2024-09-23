# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install OpenGL library and the dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libgl1 && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 7002

# Set environment variables
ENV FACELANDMARKS_DEBUG=false
ENV FACELANDMARKS_PORT=7002

# Run the application
CMD ["python", "src/main.py"]