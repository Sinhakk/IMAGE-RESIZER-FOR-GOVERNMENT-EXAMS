# ═══════════════════════════════════════════════════════════
#  Image Utility Bot v6.5 — Production Dockerfile
#  Base: python:3.11-slim (mediapipe 0.10.14 cp311 wheels ✓)
# ═══════════════════════════════════════════════════════════
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # glibc memory fragmentation kam karta hai — 512MB tier pe
    # PTB threads + mediapipe ke saath helpful
    MALLOC_ARENA_MAX=2 \
    PORT=8080

# Render/Platformers SIGTERM bhejte hain — explicit stop signal
STOPSIGNAL SIGTERM

WORKDIR /app

# ── System libs ──
# mediapipe opencv-contrib-python (full) laata hai — import ke waqt
# libGL + glib chahiye. GUI use nahi hota, sirf import ke liye.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies pehle copy (layer cache — code change pe
#    heavy pip install SKIP hoga, rebuild seconds mein) ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──
# ⚠️ File ka naam bot.py hona chahiye — kuch aur hai to yahan + CMD update karo
COPY bot.py .

# ── Security: root ke roop mein mat chalao ──
RUN useradd --create-home botuser
USER botuser

EXPOSE 8080

# ── Health check — /health endpoint (start-period: mediapipe warm-up ke liye) ──
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/health', timeout=4)" || exit 1

CMD ["python", "bot.py"]
