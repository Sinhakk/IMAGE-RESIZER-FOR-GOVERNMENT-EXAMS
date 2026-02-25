# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the bot script
COPY bot.py .

# Set environment variable for bot token (must be provided at runtime)
ENV BOT_TOKEN=""

# Run the bot
CMD ["python", "bot.py"]
