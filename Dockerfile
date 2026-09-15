# ═══════════════════════════════════════════════════════════
#  Image Utility Bot v6.0 — Production Dockerfile
#  Base: python:3.11-slim (mediapipe 0.10.14 cp311 wheels ✓)
# ═══════════════════════════════════════════════════════════
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # glibc memory fragmentation kam karta hai — Render free
    # tier (512MB) pe PTB threads + mediapipe ke saath helpful
    MALLOC_ARENA_MAX=2 \
    PORT=8080

WORKDIR /app

# ── System libs ──
# mediapipe opencv-contrib-python (full) laata hai — import ke
# waqt libGL + glib chahiye. Bot GUI use nahi karta, sirf
# import ke liye. (headless conflict se bachne ke liye full hi rakhna safe hai)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies pehle copy (layer cache — code change pe
#    heavy pip install SKIP hoga, rebuild seconds mein) ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──
# ⚠️ Script ka naam bot.py hona chahiye — agar kuch aur hai
#    to neeche CMD bhi update karna
COPY bot.py .

# ── Security: root ke roop mein mat chalao ──
RUN useradd --create-home botuser
USER botuser

EXPOSE 8080

# ── Health check — Flask /health endpoint (self-healing ke liye) ──
# start-period 90s: mediapipe warm-up ke liye time
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/health', timeout=4)" || exit 1

CMD ["python", "bot.py"]
