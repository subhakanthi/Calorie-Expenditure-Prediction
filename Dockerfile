FROM python:3.8-slim-buster

# Set working directory
WORKDIR /app

# Copy app code
COPY . /app

# Install OS-level dependencies (excluding awscli)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 unzip && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run the Flask application
CMD ["python3", "application.py"]
