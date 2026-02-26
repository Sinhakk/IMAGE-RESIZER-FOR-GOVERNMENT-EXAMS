# ─────────────────────────────────────────────
# Base Image (Lightweight)
# ─────────────────────────────────────────────
FROM python:3.11-slim

# ─────────────────────────────────────────────
# Prevent Python from buffering stdout/stderr
# ─────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ─────────────────────────────────────────────
# Install system dependencies required by:
# - opencv
# - mediapipe
# - pillow
# ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libgtk-3-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────
# Create non-root user (security)
# ─────────────────────────────────────────────
RUN useradd -m botuser
WORKDIR /app

# ─────────────────────────────────────────────
# Copy requirements first (better caching)
# ─────────────────────────────────────────────
COPY requirements.txt .

# Upgrade pip + install deps
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# Copy project files
# ─────────────────────────────────────────────
COPY . .

# Change ownership
RUN chown -R botuser:botuser /app
USER botuser

# ─────────────────────────────────────────────
# Expose port for Flask health server
# Render will provide PORT env variable
# ─────────────────────────────────────────────
EXPOSE 8080

# ─────────────────────────────────────────────
# Start bot
# ─────────────────────────────────────────────
CMD ["python", "bot.py"]
