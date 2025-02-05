FROM python:3.13-slim

WORKDIR /app

# Install system dependencies and uv in the same layer
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && . "${HOME}/.local/bin/env"

# Copy project files
COPY pyproject.toml .
COPY uv.lock .
COPY yt-dlp.sh .
COPY entrypoint.sh /entrypoint.sh

# uv stuff
RUN . "${HOME}/.local/bin/env" \
    && uv venv .venv \
    && . .venv/bin/activate \
    && uv pip install -e . \
    # Install yt-dlp
    && uv pip install -U --pre yt-dlp \
    # Create update script
    && chmod +x /entrypoint.sh \
    && chmod +x /app/yt-dlp.sh \
    && echo '*/10 * * * * /app/yt-dlp.sh' | crontab -

# Copy application code
COPY . .

# Create downloads directory
RUN mkdir -p src/downloads

# Expose the port the app runs on
EXPOSE 7005

# Set environment variables
ENV YOUTUBEDL_DEBUG=false \
    YOUTUBEDL_PORT=7005 \
    R2_ENDPOINT="" \
    R2_BUCKET="" \
    R2_PUBLIC_URL="" \
    PATH="/app/.venv/bin:${PATH}"

# Run the app using the Python from the virtual environment
ENTRYPOINT ["/entrypoint.sh"]