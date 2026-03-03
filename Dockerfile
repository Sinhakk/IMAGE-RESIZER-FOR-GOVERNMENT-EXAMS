# ─────────────────────────────────────────────────────────────────────
# IMAGE UTILITY BOT — Dockerfile
# Base: python:3.11-slim (mediapipe needs 3.11 max, not 3.12)
# ─────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps:
#  libgl1 + libglib2.0-0  → OpenCV (headless still needs these)
#  libgomp1               → mediapipe (OpenMP)
#  libsm6 + libxext6      → some PIL/cv2 edge cases
#  wget + ca-certificates → mediapipe model download at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — never run bots as root
RUN useradd -m -u 1000 botuser

WORKDIR /app

# Install Python deps first (layer cache — rebuild only if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy bot code
COPY main.py .

# DB lives in /tmp (Render ephemeral disk) — writable by botuser
RUN mkdir -p /tmp && chown botuser:botuser /tmp

USER botuser

# Port for Flask health check (Render uses this)
EXPOSE 8080

# Entrypoint
CMD ["python", "main.py"]
