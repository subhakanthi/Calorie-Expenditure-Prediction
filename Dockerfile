FROM python:3.8-slim-buster

# Set working directory
WORKDIR /app

# Copy app code
COPY . /app

# Install OS-level dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    awscli ffmpeg libsm6 libxext6 unzip && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port (not strictly required by EB but good practice)
EXPOSE 5000

# Set the command to run your app
CMD ["python3", "application.py"]
