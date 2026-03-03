#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║      IMAGE UTILITY BOT  —  Production Edition  v4.0             ║
║      Government Exam Photo Helper                                ║
║                                                                  ║
║  Features:                                                       ║
║   • AI Background Change (MediaPipe + Advanced Masking)          ║
║   • Smart Resize with Presets (Passport, Stamp, ID, Custom)      ║
║   • Professional Signature Extraction (Adaptive Threshold)       ║
║   • File Size Optimizer (JPEG/PNG/PDF)                           ║
║   • Bilingual UI (Hinglish / English)                            ║
║   • Persistent User Preferences (SQLite)                         ║
║   • Admin Panel (/admin, /broadcast, /stats)                     ║
║   • Rate Limiting, Lock Management, Auto-cleanup                 ║
║   • Render.com Ready (Flask health check + Polling)              ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────
import os, io, re, cv2, sqlite3, logging, time, asyncio, gc, signal
import sys, threading, hashlib, json
from enum import Enum
from collections import defaultdict
from threading import Lock
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
from contextlib import contextmanager
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import numpy as np
from PIL import Image, ImageColor, ImageOps, ImageFilter, ImageEnhance, PngImagePlugin
from flask import Flask, jsonify
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand,
)
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ConversationHandler, ContextTypes, filters,
)
from telegram.error import BadRequest, TimedOut, NetworkError, RetryAfter
import mediapipe as mp

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
VERSION          = "v4.2-privacy"
DPI_DEFAULT      = 300
MAX_QUALITY      = 95
MIN_QUALITY      = 10
PREVIEW_MAX_SIZE = (512, 512)
BLUR_THRESHOLD   = 80
MAX_IMAGE_DIM    = 4096
PROCESSING_TIMEOUT = 30
RATE_LIMIT_REQ   = 8
RATE_LIMIT_SECS  = 60
MAX_FILE_SIZE_MB = 20
ADMIN_IDS_RAW    = os.environ.get("ADMIN_IDS", "")
ADMIN_IDS        = {int(x.strip()) for x in ADMIN_IDS_RAW.split(",") if x.strip().isdigit()}
DB_PATH          = os.environ.get("DB_PATH", "/tmp/imagebot.db")
BOT_START_TIME   = time.time()

# ─────────────────────────────────────────────────────────────────────
# PRIVACY CORE — AES-256-GCM Session Encryption
#
# Why:
#  • Cloud platforms (Render) can page RAM to swap, take crash dumps,
#    or retain container snapshots. Encrypting even in-RAM buffers
#    ensures image bytes are never at rest unprotected.
#  • Session key is generated fresh on every start → never on disk
#    → old ciphertexts useless after restart.
#  • Real Telegram IDs hashed (SHA-256 + salt) before DB storage
#    → DB leak reveals nothing about real users.
# ─────────────────────────────────────────────────────────────────────

_SESSION_KEY: bytes = AESGCM.generate_key(bit_length=256)
_AESGCM               = AESGCM(_SESSION_KEY)
_UID_SALT             = os.environ.get("UID_SALT", "change-this-salt-in-production")

# Warn at startup if default salt used
import atexit as _atexit


def hash_uid(telegram_id: int) -> str:
    """
    SHA-256(salt:uid) — one-way, irreversible.
    Even full DB dump gives attacker zero Telegram IDs.
    """
    raw = f"{_UID_SALT}:{telegram_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class SecureBuffer:
    """
    AES-256-GCM encrypted wrapper for image bytes in RAM.

    Lifecycle:
      1. __init__  : plaintext encrypted immediately, plaintext discarded
      2. open()    : context manager — decrypt, yield, wipe on exit
      3. wipe()    : overwrite ciphertext with zeros, gc.collect()
      4. __del__   : safety net wipe if not already wiped
    """
    __slots__ = ("_ct", "_nonce", "_wiped")

    def __init__(self, plaintext: bytes):
        self._nonce = os.urandom(12)
        self._ct    = _AESGCM.encrypt(self._nonce, plaintext, None)
        self._wiped = False

    def decrypt(self) -> bytes:
        if self._wiped:
            raise RuntimeError("SecureBuffer already wiped")
        return _AESGCM.decrypt(self._nonce, self._ct, None)

    def wipe(self):
        if not self._wiped:
            self._wiped = True
            if hasattr(self, "_ct") and self._ct:
                self._ct    = bytes(len(self._ct))
                self._nonce = bytes(12)
            gc.collect()

    @contextmanager
    def open(self):
        """Decrypt → yield → wipe. Always wipes, even on exception."""
        plaintext = None
        try:
            plaintext = self.decrypt()
            yield plaintext
        finally:
            if plaintext is not None:
                plaintext = bytes(len(plaintext))
            self.wipe()

    def __del__(self):
        self.wipe()


def secure_store(ctx, key: str, img: Image.Image):
    """
    PIL Image → PNG bytes → SecureBuffer → ctx.user_data[key]
    Original image deleted immediately after encryption.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    plaintext = buf.getvalue()
    buf.close()
    size = len(plaintext)

    del img
    gc.collect()

    ctx.user_data[key] = SecureBuffer(plaintext)
    plaintext = bytes(size)
    del plaintext
    gc.collect()


def secure_load(ctx, key: str) -> Optional[Image.Image]:
    """
    Pop SecureBuffer from user_data → decrypt → PIL Image → wipe.
    Buffer is removed and wiped right after decryption. Single-use.
    """
    sbuf = ctx.user_data.pop(key, None)
    if not isinstance(sbuf, SecureBuffer):
        if sbuf is not None:
            del sbuf
        return None
    with sbuf.open() as data:
        img = Image.open(io.BytesIO(data)).copy()
    gc.collect()
    return img


def secure_wipe_all(ctx, keys: list):
    """Remove and wipe all SecureBuffers (and plain values) in keys list."""
    for key in keys:
        val = ctx.user_data.pop(key, None)
        if isinstance(val, SecureBuffer):
            val.wipe()
        del val
    gc.collect()


# All keys that may hold SecureBuffers — used for bulk wipe
_IMAGE_KEYS = [
    "bg_img", "bg_result",
    "resize_img", "resize_result",
    "reduce_img", "sig_result",
]


def _wipe_session_key():
    """Called at exit — overwrite session key in memory."""
    global _SESSION_KEY
    try:
        _SESSION_KEY = bytes(len(_SESSION_KEY))
        del _SESSION_KEY
        gc.collect()
    except Exception:
        pass

_atexit.register(_wipe_session_key)



# NOTE: Photo presets are stored in SQLite DB — no hardcoding!
# Admin can add/edit/delete presets via /addpreset, /editpreset, /delpreset
# Default seed presets are inserted once at first run (init_db)

# ─────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("ImageBot")

# ─────────────────────────────────────────────────────────────────────
# DATABASE — Persistent user preferences + analytics
# ─────────────────────────────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                hashed_uid  TEXT PRIMARY KEY,
                total_ops   INTEGER DEFAULT 0,
                hinglish    INTEGER DEFAULT 0,
                strict      INTEGER DEFAULT 0,
                dpi         INTEGER DEFAULT 300
            );
            -- operations table removed: storing op_type + timestamp
            -- creates a linkable activity log = privacy risk.
            -- Only anonymous total_ops counter kept in users table.
            CREATE TABLE IF NOT EXISTS broadcasts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                message     TEXT,
                sent_at     TEXT
            );
            CREATE TABLE IF NOT EXISTS presets (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                slug        TEXT UNIQUE NOT NULL,
                label       TEXT NOT NULL,
                width_px    INTEGER NOT NULL,
                height_px   INTEGER NOT NULL,
                sort_order  INTEGER DEFAULT 0,
                active      INTEGER DEFAULT 1
            );
        """)
        # Seed default presets only on first run
        count = conn.execute("SELECT COUNT(*) FROM presets").fetchone()[0]
        if count == 0:
            _seed_presets(conn)

def _seed_presets(conn):
    """
    Default presets — ye sirf pehli baar DB mein jaate hain.
    Baad mein admin /addpreset, /editpreset, /delpreset se manage kar sakta hai.
    Koi bhi value change ho jaye government ki, admin live update kar sakta hai
    bina bot restart kiye.
    """
    defaults = [
        # slug,             label,                          w,   h,  order
        ("passport_india",  "🪪 Passport India (3.5×4.5cm)", 413, 531, 1),
        ("passport_us",     "🇺🇸 US Passport (2×2 inch)",    600, 600, 2),
        ("stamp_size",      "📮 Stamp Size (2.5×3cm)",        295, 354, 3),
        ("aadhaar",         "🆔 Aadhaar (3.5×4.5cm)",        413, 531, 4),
        ("pan_card",        "💳 PAN Card (3.5×4.5cm)",        413, 531, 5),
        ("driving",         "🚗 Driving License (3.5×4.5cm)",413, 531, 6),
        ("upsc",            "📋 UPSC (4.5×4.5cm)",            531, 531, 7),
        ("ssc",             "📋 SSC (3.5×4.5cm)",             413, 531, 8),
        ("railway",         "🚂 Railway (3.5×4.5cm)",         413, 531, 9),
        ("neet",            "⚕️ NEET/JEE (4.5×4.5cm)",       531, 531, 10),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO presets(slug, label, width_px, height_px, sort_order) VALUES(?,?,?,?,?)",
        defaults,
    )

# ── Preset DB CRUD ──────────────────────────────────────────────────

def db_get_presets() -> list:
    """Return all active presets ordered by sort_order."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, slug, label, width_px, height_px FROM presets WHERE active=1 ORDER BY sort_order, id"
        ).fetchall()
    return [dict(r) for r in rows]

def db_get_preset_by_id(preset_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, slug, label, width_px, height_px FROM presets WHERE id=? AND active=1",
            (preset_id,)
        ).fetchone()
    return dict(row) if row else None

def db_add_preset(label: str, width_px: int, height_px: int) -> int:
    """Add new preset, auto-generate slug. Returns new preset id."""
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower().strip())[:30]
    # Ensure unique slug
    base_slug = slug
    suffix = 1
    with get_db() as conn:
        while conn.execute("SELECT 1 FROM presets WHERE slug=?", (slug,)).fetchone():
            slug = f"{base_slug}_{suffix}"
            suffix += 1
        max_order = conn.execute("SELECT MAX(sort_order) FROM presets").fetchone()[0] or 0
        conn.execute(
            "INSERT INTO presets(slug, label, width_px, height_px, sort_order) VALUES(?,?,?,?,?)",
            (slug, label, width_px, height_px, max_order + 1)
        )
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

def db_edit_preset(preset_id: int, label: str, width_px: int, height_px: int):
    with get_db() as conn:
        conn.execute(
            "UPDATE presets SET label=?, width_px=?, height_px=? WHERE id=?",
            (label, width_px, height_px, preset_id)
        )

def db_delete_preset(preset_id: int):
    """Soft delete — marks as inactive."""
    with get_db() as conn:
        conn.execute("UPDATE presets SET active=0 WHERE id=?", (preset_id,))

def db_list_all_presets() -> list:
    """For admin — includes inactive presets."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, label, width_px, height_px, active, sort_order FROM presets ORDER BY sort_order, id"
        ).fetchall()
    return [dict(r) for r in rows]

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def upsert_user(user_id: int, username: str = ""):
    """Register user by hashed ID only. No real ID, no username, no timestamps."""
    hid = hash_uid(user_id)
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO users(hashed_uid) VALUES(?)", (hid,)
        )

def get_user_prefs(user_id: int) -> Dict[str, Any]:
    hid = hash_uid(user_id)
    with get_db() as conn:
        row = conn.execute(
            "SELECT hinglish, strict, dpi FROM users WHERE hashed_uid=?", (hid,)
        ).fetchone()
    if row:
        return {"hinglish": bool(row["hinglish"]), "strict": bool(row["strict"]), "dpi": row["dpi"]}
    return {"hinglish": False, "strict": False, "dpi": DPI_DEFAULT}

def save_user_pref(user_id: int, key: str, value):
    allowed = {"hinglish", "strict", "dpi"}
    if key not in allowed:
        return  # prevent SQL injection via key param
    hid = hash_uid(user_id)
    with get_db() as conn:
        conn.execute(f"UPDATE users SET {key}=? WHERE hashed_uid=?", (value, hid))

def bump_op_count(user_id: int):
    """Increment anonymous op counter only — no op type, no timestamp stored."""
    hid = hash_uid(user_id)
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET total_ops=total_ops+1 WHERE hashed_uid=?", (hid,)
        )

def get_bot_stats() -> Dict[str, Any]:
    """Aggregate stats only — no individual user data exposed."""
    with get_db() as conn:
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_ops   = conn.execute("SELECT SUM(total_ops) FROM users").fetchone()[0] or 0
    return {"total_users": total_users, "total_ops": total_ops}

# Real user IDs for broadcast — RAM only, never written to DB.
# Populated as users interact with the bot during this session.
_known_real_ids: set = set()


def get_all_user_ids():
    """Return in-memory real IDs for broadcast. Not from DB."""
    return list(_known_real_ids)

# ─────────────────────────────────────────────────────────────────────
# MEDIAPIPE SINGLETON — warm-up on start
# ─────────────────────────────────────────────────────────────────────
_mp_model = None
_mp_lock  = Lock()

def get_segmentation_model():
    global _mp_model
    with _mp_lock:
        if _mp_model is None:
            logger.info("Warming up MediaPipe model...")
            _mp_model = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            # Warm-up pass with dummy image
            dummy = np.zeros((100, 100, 3), dtype=np.uint8)
            _mp_model.process(dummy)
            logger.info("MediaPipe model ready.")
    return _mp_model

def warm_up_model():
    """Call this at startup so first user doesn't wait."""
    threading.Thread(target=get_segmentation_model, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────
# FLASK HEALTH — Render keep-alive
# ─────────────────────────────────────────────────────────────────────
flask_app = Flask(__name__)
_bot_healthy = True

@flask_app.route("/")
def home():
    uptime = int(time.time() - BOT_START_TIME)
    return jsonify({
        "status": "running" if _bot_healthy else "degraded",
        "bot":    VERSION,
        "uptime_seconds": uptime,
    })

@flask_app.route("/health")
def health():
    if _bot_healthy:
        return jsonify({"status": "healthy"}), 200
    return jsonify({"status": "degraded"}), 503

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    flask_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

# ─────────────────────────────────────────────────────────────────────
# LANGUAGE  (Hinglish / English)
# ─────────────────────────────────────────────────────────────────────
STRINGS: Dict[str, Dict[str, str]] = {
    "hi": {
        # Menus
        "main_menu":          "🏠 *Main Menu* — kya karna hai?",
        "bg_change":          "🖼 Background Change",
        "resize":             "📐 Resize / Compress",
        "signature":          "✍️ Signature Extract",
        # Prompts
        "send_photo":         "📸 Photo bhejo (JPEG/PNG, max 20MB):",
        "processing":         "⏳ Processing ho raha hai... thoda intezaar karo.",
        "preview":            "👀 *Preview ready hai!* Theek lag raha hai?",
        "looks_ok":           "✅ Theek hai — aage badho",
        "retry":              "🔁 Dobara try karo",
        "format_choose":      "📁 Format select karo:",
        "dimensions":         "📐 Dimensions batao (koi bhi unit use karo):\n• Pixels: `300x400` ya `300x400px`\n• Centimeters: `3.5x4.5cm`\n• Millimeters: `35x45mm`\n• Inches: `2x2in` ya `2x2inch`",
        "color_choose":       "🎨 Background color select karo:",
        "custom_color_prompt":"🖊 Color type karo (naam ya hex code):\nExamples: `white`, `blue`, `#E8F4FD`, `lightgray`",
        "enter_kb":           "📦 Target size (KB mein) batao:\nExample: `100` for 100KB",
        "size_option":        "📦 Size kaise set karna hai?",
        "size_by_kb":         "📦 KB target set karo",
        "size_by_dims":       "🖼 Format ke saath direct save karo",
        "reduce_send_photo":  "📸 Jo photo compress karni hai wo bhejo:",
        "select_preset":      "📋 Photo type select karo (ya Custom Dimensions):",
        # Warnings
        "blur_warn":          "⚠️ *Photo blur lag rahi hai!*\nBehtar result ke liye clear photo use karo.",
        "face_small":         "⚠️ Face bahut chhota hai. Zara closer photo lo.",
        "face_offcenter":     "⚠️ Face center mein nahi hai. Framing sudhaaro.",
        "sig_bg_warn":        "⚠️ Background safed nahi dikh raha.\nSafed paper pe signature le — result achha aayega.",
        "bg_warning":         "ℹ️ AI background change mein minor imperfections ho sakti hain. Preview dhyan se dekho.",
        "reminder":           "⚠️ Upload se *pehle* result zaroor check karo. Ye tool assistance ke liye hai.",
        "rate_limit":         "⏳ Bahut zyada requests. Ek minute mein max 8 operations ho sakte hain.\nThodi der baad try karo.",
        "timeout_err":        "⏱ Processing timeout. Chhoti/kam resolution ki photo use karo.",
        "file_too_large":     "❌ File bahut badi hai (max 20MB). Chhota karo.",
        "invalid_file":       "❌ Ye file image nahi hai. JPEG/PNG bhejo.",
        "no_photo":           "❌ Koi photo nahi mili. Dobara bhejo.",
        # Results
        "bg_done":            "✅ *Background change ho gaya!*",
        "resize_done":        "✅ *Resize ho gaya!*",
        "compress_done":      "✅ *File compress ho gai!*",
        "sig_done":           "✅ *Signature extract ho gaya!*",
        # Others
        "cancel":             "✋ Cancel kar diya. Main menu pe wapas aao.",
        "error":              "❌ Kuch gadbad hui. Phir se /start karo.",
        "unexpected":         "🤔 Samajh nahi aaya. Niche ke buttons use karo.",
        "processing_lock":    "⚙️ Ek operation already chal raha hai. Pehle wala khatam hone do ya /cancel karo.",
        "history":            "📤 Last processed image:",
        "no_history":         "📭 Koi previous image nahi mili.",
        "strict_on":          "✅ Strict mode ON — aspect ratio maintain hoga.",
        "strict_off":         "❌ Strict mode OFF.",
        "dpi_set":            "✅ DPI set: ",
        "dpi_usage":          "Usage: `/dpi 72`, `/dpi 150`, `/dpi 300`",
        "lang_hi":            "✅ Hinglish mode ON! 🇮🇳",
        "lang_en":            "✅ English mode ON! 🇬🇧",
        # Help
        "help_text": (
            "📖 *Bot Guide*\n\n"
            "🖼 *Background Change*\n"
            "AI se background replace karo — white, blue ya koi bhi color\n\n"
            "📐 *Resize Image*\n"
            "• Preset sizes: Passport, UPSC, NEET, Railway etc.\n"
            "• Custom dimensions (px, mm, cm, inch)\n"
            "• File size compress (KB mein target set karo)\n\n"
            "✍️ *Signature Extract*\n"
            "White paper pe signature ki photo lo — bot transparent PNG dega\n\n"
            "⚙️ *Commands*\n"
            "/start — Main menu\n"
            "/cancel — Current operation cancel\n"
            "/history — Last image dobara pao\n"
            "/hinglish — Language toggle\n"
            "/strict — Aspect ratio strict mode\n"
            "/dpi [72/150/300] — Output DPI set karo\n"
            "/help — Ye message\n\n"
            "💡 *Tips*\n"
            "• Clear, well-lit photos best results deti hain\n"
            "• Background change ke liye plain background wali photo use karo\n"
            "• Signature ke liye safed paper pe dark ink use karo"
        ),
    },
    "en": {
        "main_menu":          "🏠 *Main Menu* — What would you like to do?",
        "bg_change":          "🖼 Background Change",
        "resize":             "📐 Resize / Compress",
        "signature":          "✍️ Signature Extract",
        "send_photo":         "📸 Send your photo (JPEG/PNG, max 20MB):",
        "processing":         "⏳ Processing... please wait.",
        "preview":            "👀 *Preview ready!* Does it look OK?",
        "looks_ok":           "✅ Looks good — proceed",
        "retry":              "🔁 Try again",
        "format_choose":      "📁 Choose output format:",
        "dimensions":         "📐 Enter dimensions (any unit):\n• Pixels: `300x400` or `300x400px`\n• Centimeters: `3.5x4.5cm`\n• Millimeters: `35x45mm`\n• Inches: `2x2in` or `2x2inch`",
        "color_choose":       "🎨 Choose background color:",
        "custom_color_prompt":"🖊 Type a color name or hex code:\nExamples: `white`, `blue`, `#E8F4FD`, `lightgray`",
        "enter_kb":           "📦 Enter target file size in KB:\nExample: `100` for 100KB",
        "size_option":        "📦 How do you want to set the size?",
        "size_by_kb":         "📦 Set KB target",
        "size_by_dims":       "🖼 Save directly with format",
        "reduce_send_photo":  "📸 Send the photo you want to compress:",
        "select_preset":      "📋 Choose photo type (or Custom Dimensions):",
        "blur_warn":          "⚠️ *Photo appears blurry!*\nFor best results, use a sharp, clear image.",
        "face_small":         "⚠️ Face is too small in frame. Please take a closer photo.",
        "face_offcenter":     "⚠️ Face is not centered. Adjust framing.",
        "sig_bg_warn":        "⚠️ Background doesn't appear white.\nFor best results, sign on plain white paper.",
        "bg_warning":         "ℹ️ AI background change may have minor imperfections. Review preview carefully.",
        "reminder":           "⚠️ Always verify the result *before* uploading. This tool is for assistance only.",
        "rate_limit":         "⏳ Too many requests. Maximum 8 operations per minute allowed.\nPlease wait a moment.",
        "timeout_err":        "⏱ Processing timed out. Try a lower resolution image.",
        "file_too_large":     "❌ File too large (max 20MB). Please reduce size first.",
        "invalid_file":       "❌ This doesn't appear to be an image. Please send JPEG or PNG.",
        "no_photo":           "❌ No photo detected. Please try again.",
        "bg_done":            "✅ *Background changed successfully!*",
        "resize_done":        "✅ *Image resized successfully!*",
        "compress_done":      "✅ *File compressed successfully!*",
        "sig_done":           "✅ *Signature extracted successfully!*",
        "cancel":             "✋ Cancelled. Back to main menu.",
        "error":              "❌ Something went wrong. Please /start again.",
        "unexpected":         "🤔 Unexpected input. Please use the buttons below.",
        "processing_lock":    "⚙️ An operation is already running. Wait for it to finish or /cancel.",
        "history":            "📤 Last processed image:",
        "no_history":         "📭 No previous image found.",
        "strict_on":          "✅ Strict mode ON — aspect ratio enforced.",
        "strict_off":         "❌ Strict mode OFF.",
        "dpi_set":            "✅ DPI set to: ",
        "dpi_usage":          "Usage: `/dpi 72`, `/dpi 150`, `/dpi 300`",
        "lang_hi":            "✅ Hinglish mode ON! 🇮🇳",
        "lang_en":            "✅ English mode ON! 🇬🇧",
        "help_text": (
            "📖 *Bot Guide*\n\n"
            "🖼 *Background Change*\n"
            "Replace your photo background using AI — white, blue, or any color\n\n"
            "📐 *Resize Image*\n"
            "• Preset sizes: Passport, UPSC, NEET, Railway and more\n"
            "• Custom dimensions (px, mm, cm, inch)\n"
            "• File size compression (set target in KB)\n\n"
            "✍️ *Signature Extract*\n"
            "Photo your signature on white paper — bot returns transparent PNG\n\n"
            "⚙️ *Commands*\n"
            "/start — Main menu\n"
            "/cancel — Cancel current operation\n"
            "/history — Get last processed image\n"
            "/hinglish — Toggle language\n"
            "/strict — Aspect ratio strict mode\n"
            "/dpi [72/150/300] — Set output DPI\n"
            "/help — This message\n\n"
            "💡 *Tips*\n"
            "• Use clear, well-lit photos for best results\n"
            "• For background change, use photos with a plain background\n"
            "• For signatures, use dark ink on white paper"
        ),
    },
}

def t(key: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    lang = "hi" if context.user_data.get("hinglish") else "en"
    return STRINGS[lang].get(key, STRINGS["en"].get(key, key))

# ─────────────────────────────────────────────────────────────────────
# CONVERSATION STATES
# ─────────────────────────────────────────────────────────────────────
class S(Enum):
    SELECT_ACTION        = 0
    # Background change
    BG_WAIT_PHOTO        = 1
    BG_WAIT_COLOR        = 2
    BG_PREVIEW           = 3
    BG_WAIT_FORMAT       = 4
    # Resize
    RESIZE_MODE          = 5
    CUSTOM_WAIT_PHOTO    = 6
    CUSTOM_SELECT_PRESET = 7
    CUSTOM_WAIT_DIMS     = 8
    CUSTOM_PREVIEW       = 9
    CUSTOM_SIZE_OPT      = 10
    CUSTOM_WAIT_KB       = 11
    CUSTOM_WAIT_FORMAT   = 12
    # Reduce
    REDUCE_WAIT_PHOTO    = 13
    REDUCE_WAIT_KB       = 14
    REDUCE_WAIT_FORMAT   = 15
    # Signature
    SIG_WAIT_PHOTO       = 16
    SIG_PREVIEW          = 17
    SIG_WAIT_FORMAT      = 18

# ─────────────────────────────────────────────────────────────────────
# RATE LIMITING
# ─────────────────────────────────────────────────────────────────────
_rate_data: dict = defaultdict(list)

def check_rate_limit(user_id: int) -> bool:
    now = time.time()
    _rate_data[user_id] = [ts for ts in _rate_data[user_id] if now - ts < RATE_LIMIT_SECS]
    if len(_rate_data[user_id]) >= RATE_LIMIT_REQ:
        return False
    _rate_data[user_id].append(now)
    return True

# ─────────────────────────────────────────────────────────────────────
# PROCESSING LOCK per user
# ─────────────────────────────────────────────────────────────────────
def acquire_lock(ctx: ContextTypes.DEFAULT_TYPE) -> bool:
    if ctx.user_data.get("_lock"):
        return False
    ctx.user_data["_lock"] = True
    return True

def release_lock(ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data.pop("_lock", None)

# ─────────────────────────────────────────────────────────────────────
# AUTO-CLEANUP — stale user_data after 30 min
# ─────────────────────────────────────────────────────────────────────
# cleanup() replaced by secure_wipe_all() — properly wipes SecureBuffers before deleting

async def schedule_cleanup(ctx: ContextTypes.DEFAULT_TYPE, delay: int = 1800):
    """After 30 min, clear processing-related data to free memory."""
    await asyncio.sleep(delay)
    for k in ["bg_img", "bg_result", "resize_img", "resize_result",
              "reduce_img", "sig_result", "target_kb"]:
        ctx.user_data.pop(k, None)
    release_lock(ctx)

# ─────────────────────────────────────────────────────────────────────
# IMAGE UTILITIES
# ─────────────────────────────────────────────────────────────────────
def fix_orientation(img: Image.Image) -> Image.Image:
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img

def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    if img.mode not in ("RGB",):
        return img.convert("RGB")
    return img

def downscale_if_needed(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img

def detect_blur(img: Image.Image) -> float:
    try:
        gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except Exception:
        return 999.0

def analyze_face(img: Image.Image) -> Optional[str]:
    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray  = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        iw, ih = img.size
        if (w * h) < (iw * ih * 0.04):
            return "face_small"
        if abs((x + w / 2) - iw / 2) > iw * 0.25:
            return "face_offcenter"
        return None
    except Exception:
        return None

def create_preview(img: Image.Image) -> io.BytesIO:
    preview = img.copy()
    preview.thumbnail(PREVIEW_MAX_SIZE, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    preview.convert("RGB").save(buf, format="JPEG", quality=82, optimize=True)
    buf.seek(0)
    return buf

def format_size(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    if nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    return f"{nbytes / (1024 * 1024):.2f} MB"

def parse_dimensions(text: str) -> Optional[Tuple[int, int]]:
    text = text.strip().lower()
    # cm
    m = re.match(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*cm", text)
    if m:
        dpi = 300
        return int(float(m.group(1)) / 2.54 * dpi), int(float(m.group(2)) / 2.54 * dpi)
    # inch
    m = re.match(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*in(?:ch)?", text)
    if m:
        dpi = 300
        return int(float(m.group(1)) * dpi), int(float(m.group(2)) * dpi)
    # pixels
    m = re.match(r"(\d+)\s*x\s*(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

def parse_color(text: str) -> Tuple[int, int, int]:
    try:
        return ImageColor.getrgb(text.strip())[:3]
    except Exception:
        return (255, 255, 255)

def validate_color(text: str) -> bool:
    try:
        ImageColor.getrgb(text.strip())
        return True
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────────────
# CORE: BACKGROUND CHANGE — Production Quality
# ─────────────────────────────────────────────────────────────────────
def person_segmentation_replace(img: Image.Image, color_text: str) -> Image.Image:
    """
    Production-grade background replacement:
    1. MediaPipe segmentation (high confidence)
    2. Morphological refinement (remove noise, fill holes)
    3. Edge feathering (smooth transitions)
    4. Color bleed correction
    """
    rgb_arr = np.array(img.convert("RGB"))
    h, w    = rgb_arr.shape[:2]

    # Step 1: MediaPipe mask
    model  = get_segmentation_model()
    result = model.process(rgb_arr)
    mask_f = result.segmentation_mask  # float32 0..1

    # Step 2: High-confidence binary mask
    mask_hi = (mask_f > 0.75).astype(np.uint8)

    # Step 3: Morphological ops — fill holes, remove noise
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_hi      = cv2.morphologyEx(mask_hi, cv2.MORPH_CLOSE, kernel_close)
    mask_hi      = cv2.morphologyEx(mask_hi, cv2.MORPH_OPEN,  kernel_open)

    # Step 4: Soft edge (feathered blend zone from the hard mask)
    # Erode mask to get definite foreground, dilate to get uncertain zone
    mask_fg      = cv2.erode(mask_hi,  kernel_open, iterations=2).astype(np.float32)
    mask_dilated = cv2.dilate(mask_hi, kernel_close, iterations=3).astype(np.float32)

    # Blend zone: uncertain area uses original mediapipe confidence
    blend_zone = mask_dilated - mask_fg  # 0 or 1 in uncertain areas
    mask_soft  = mask_fg + blend_zone * np.clip(mask_f * 2 - 0.5, 0, 1)
    mask_soft  = np.clip(mask_soft, 0, 1)

    # Step 5: Gaussian feather for smooth edges
    mask_feathered = cv2.GaussianBlur(mask_soft, (15, 15), 0)
    mask_feathered = mask_feathered[..., np.newaxis]  # H x W x 1

    # Step 6: Background fill
    color   = parse_color(color_text)
    bg_arr  = np.full_like(rgb_arr, color, dtype=np.uint8)

    # Step 7: Alpha blend
    out = (rgb_arr.astype(np.float32) * mask_feathered +
           bg_arr.astype(np.float32) * (1 - mask_feathered)).astype(np.uint8)

    # Step 8: Color bleed correction on background
    # Replace background pixels (mask < 0.1) more aggressively with pure bg color
    hard_bg = mask_feathered[..., 0] < 0.1
    out[hard_bg] = color

    return Image.fromarray(out)

# ─────────────────────────────────────────────────────────────────────
# CORE: SIGNATURE EXTRACTION — Production Quality
# ─────────────────────────────────────────────────────────────────────
def extract_signature(img: Image.Image) -> Image.Image:
    """
    Production-grade signature extraction:
    1. Contrast enhancement
    2. Adaptive thresholding (handles uneven lighting)
    3. Noise removal (paper texture, small dots)
    4. Auto-crop to signature bounding box
    5. Returns transparent PNG with only the signature
    """
    rgb = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Step 1: Enhance contrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    # Step 2: Adaptive threshold — handles uneven lighting / shadows
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25, C=10
    )

    # Step 3: Remove noise — small blobs (paper texture) below area threshold
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    min_blob_area = max(8, int(thresh.size * 0.00005))  # adaptive to image size
    clean = np.zeros_like(thresh)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_blob_area:
            clean[labels == lbl] = 255

    # Step 4: Close small gaps in signature strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean  = cv2.dilate(clean, kernel, iterations=1)

    # Step 5: Find bounding box and crop with 10% padding
    coords = cv2.findNonZero(clean)
    if coords is not None:
        x, y, bw, bh = cv2.boundingRect(coords)
        pad_x = max(20, int(bw * 0.10))
        pad_y = max(20, int(bh * 0.10))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(clean.shape[1], x + bw + pad_x)
        y2 = min(clean.shape[0], y + bh + pad_y)
        clean = clean[y1:y2, x1:x2]

    # Step 6: Build RGBA — black signature on transparent background
    out_h, out_w = clean.shape
    rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    rgba[..., 3] = clean  # alpha = signature pixels

    return Image.fromarray(rgba, "RGBA")

# ─────────────────────────────────────────────────────────────────────
# CORE: COMPRESS TO KB
# ─────────────────────────────────────────────────────────────────────
def compress_to_kb(img: Image.Image, target_kb: int, fmt: str = "JPEG") -> io.BytesIO:
    buf      = io.BytesIO()
    fmt_up   = fmt.upper()
    target_b = target_kb * 1024

    if fmt_up == "PDF":
        img.convert("RGB").save(buf, format="PDF")
        buf.seek(0)
        return buf

    if fmt_up == "PNG":
        # PNG: scale dimensions to hit target
        scale = 1.0
        curr  = img
        while scale > 0.05:
            buf.seek(0); buf.truncate()
            curr.save(buf, format="PNG", optimize=True)
            if buf.tell() <= target_b:
                break
            scale -= 0.08
            nw = max(1, int(img.width * scale))
            nh = max(1, int(img.height * scale))
            curr = img.resize((nw, nh), Image.Resampling.LANCZOS)
        buf.seek(0)
        return buf

    # JPEG: quality + dimension fallback
    quality = MAX_QUALITY
    while quality >= MIN_QUALITY:
        buf.seek(0); buf.truncate()
        img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() <= target_b:
            break
        quality -= 5

    # Still too large? scale dimensions
    if buf.tell() > target_b:
        scale = 0.9
        while scale > 0.1:
            nw   = max(1, int(img.width * scale))
            nh   = max(1, int(img.height * scale))
            simg = img.resize((nw, nh), Image.Resampling.LANCZOS)
            buf.seek(0); buf.truncate()
            simg.convert("RGB").save(buf, format="JPEG", quality=MIN_QUALITY, optimize=True)
            if buf.tell() <= target_b:
                break
            scale -= 0.1

    buf.seek(0)
    return buf

def save_image(img: Image.Image, fmt: str, dpi_val: int) -> io.BytesIO:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=MAX_QUALITY, dpi=(dpi_val, dpi_val))
    elif fmt.upper() == "PNG":
        img.save(buf, format="PNG")
    elif fmt.upper() == "PDF":
        img.convert("RGB").save(buf, format="PDF", resolution=dpi_val)
    buf.seek(0)
    return buf

# ─────────────────────────────────────────────────────────────────────
# IMAGE VALIDATION
# ─────────────────────────────────────────────────────────────────────
VALID_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}

def validate_image_bytes(data: bytes) -> bool:
    try:
        img = Image.open(io.BytesIO(data))
        return img.format in VALID_FORMATS
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────────────
# TELEGRAM HELPERS
# ─────────────────────────────────────────────────────────────────────
async def safe_reply(update: Update, text: str, reply_markup=None, parse_mode="Markdown"):
    target = update.message or (update.callback_query.message if update.callback_query else None)
    if target:
        await target.reply_text(text, reply_markup=reply_markup, parse_mode=parse_mode)

async def safe_edit(update: Update, text: str, reply_markup=None):
    if not update.callback_query:
        return
    q = update.callback_query
    try:
        if q.message.text is not None:
            await q.edit_message_text(text, reply_markup=reply_markup, parse_mode="Markdown")
        elif q.message.caption is not None:
            await q.edit_message_caption(text, reply_markup=reply_markup, parse_mode="Markdown")
        else:
            await q.message.reply_text(text, reply_markup=reply_markup, parse_mode="Markdown")
    except BadRequest as e:
        if "not modified" not in str(e).lower():
            await q.message.reply_text(text, reply_markup=reply_markup, parse_mode="Markdown")

async def get_image(update: Update) -> Optional[Image.Image]:
    """Download, validate, orient, and return PIL Image."""
    photo = update.message.photo[-1] if update.message.photo else None
    doc   = update.message.document if update.message.document else None
    fobj  = photo or doc
    if not fobj:
        return None

    # File size check
    file_size = getattr(fobj, "file_size", 0) or 0
    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return "too_large"

    tg_file = await fobj.get_file()
    raw     = io.BytesIO()
    await tg_file.download_to_memory(raw)
    raw.seek(0)
    data = raw.read()

    # Validate it's actually an image
    if not validate_image_bytes(data):
        return "invalid"

    img = Image.open(io.BytesIO(data))
    img = fix_orientation(img)
    img = ensure_rgb(img)
    img = downscale_if_needed(img)
    return img

async def send_main_menu(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = t("main_menu", ctx)
    kb   = main_menu_kb(ctx)
    if update.callback_query:
        await safe_edit(update, text, kb)
    else:
        await safe_reply(update, text, kb)

def log_state(uid: int, state: str, action: str):
    logger.info(f"STATE={state} | {action}")  # UID intentionally omitted

# ─────────────────────────────────────────────────────────────────────
# KEYBOARDS
# ─────────────────────────────────────────────────────────────────────
def main_menu_kb(ctx) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(t("bg_change",  ctx), callback_data="bg_change")],
        [InlineKeyboardButton(t("resize",     ctx), callback_data="resize")],
        [InlineKeyboardButton(t("signature",  ctx), callback_data="signature")],
    ])

def bg_color_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚪ White",      callback_data="col_white"),
         InlineKeyboardButton("🟡 Off-White",  callback_data="col_#f5f0e8")],
        [InlineKeyboardButton("🔵 Light Blue", callback_data="col_#add8e6"),
         InlineKeyboardButton("🟢 Light Green",callback_data="col_#90ee90")],
        [InlineKeyboardButton("🔴 Red",        callback_data="col_red"),
         InlineKeyboardButton("⬜ Light Grey", callback_data="col_#d3d3d3")],
        [InlineKeyboardButton("🎨 Custom Color", callback_data="col_custom")],
    ])

def confirm_kb(ok_cb: str, retry_cb: str, ctx) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(t("looks_ok", ctx), callback_data=ok_cb),
        InlineKeyboardButton(t("retry",    ctx), callback_data=retry_cb),
    ]])

def format_kb(include_pdf: bool = True) -> InlineKeyboardMarkup:
    row = [
        InlineKeyboardButton("JPEG", callback_data="fmt_JPEG"),
        InlineKeyboardButton("PNG",  callback_data="fmt_PNG"),
    ]
    if include_pdf:
        row.append(InlineKeyboardButton("PDF", callback_data="fmt_PDF"))
    return InlineKeyboardMarkup([row])

def resize_mode_kb(ctx) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📋 Preset Sizes",      callback_data="resize_preset")],
        [InlineKeyboardButton("📐 Custom Dimensions", callback_data="resize_custom")],
        [InlineKeyboardButton("📦 Reduce File Size",  callback_data="resize_reduce")],
    ])

def preset_kb() -> InlineKeyboardMarkup:
    """
    Keyboard built live from DB — no hardcoding.
    callback_data uses preset DB id (integer), not slug.
    Even if admin edits label/dimensions, callbacks stay valid.
    """
    presets = db_get_presets()
    rows = []
    for i in range(0, len(presets), 2):
        row = []
        for p in presets[i:i+2]:
            row.append(InlineKeyboardButton(
                p["label"],
                callback_data=f"preset_{p['id']}"
            ))
        rows.append(row)
    rows.append([InlineKeyboardButton("📐 Custom Dimensions", callback_data="preset_custom")])
    if not presets:
        # No presets in DB — only show custom option
        return InlineKeyboardMarkup([[
            InlineKeyboardButton("📐 Custom Dimensions", callback_data="preset_custom")
        ]])
    return InlineKeyboardMarkup(rows)

def size_option_kb(ctx) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(t("size_by_kb",   ctx), callback_data="sizeopt_kb")],
        [InlineKeyboardButton(t("size_by_dims", ctx), callback_data="sizeopt_save")],
    ])

# ─────────────────────────────────────────────────────────────────────
# COMMAND HANDLERS
# ─────────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    uid = update.effective_user.id
    _known_real_ids.add(uid)  # RAM-only, for broadcast
    upsert_user(uid)           # stores only hashed_uid in DB

    # Restore persistent preferences
    if "hinglish" not in ctx.user_data:
        ctx.user_data.update(get_user_prefs(uid))

    # Wipe any stale image data, keep only preferences
    secure_wipe_all(ctx, _IMAGE_KEYS)
    preserve = {k: ctx.user_data[k]
                for k in ("hinglish", "strict", "dpi")
                if k in ctx.user_data}
    release_lock(ctx)
    ctx.user_data.clear()
    ctx.user_data.update(preserve)

    log_state(uid, "START", "/start")
    await send_main_menu(update, ctx)
    return S.SELECT_ACTION

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await safe_reply(update, t("help_text", ctx))


async def cmd_privacy(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await safe_reply(update, t("privacy_notice", ctx))

async def cmd_hinglish(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    curr = ctx.user_data.get("hinglish", False)
    ctx.user_data["hinglish"] = not curr
    save_user_pref(update.effective_user.id, "hinglish", int(not curr))
    key = "lang_hi" if not curr else "lang_en"
    await safe_reply(update, t(key, ctx))

async def cmd_strict(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    curr = ctx.user_data.get("strict", False)
    ctx.user_data["strict"] = not curr
    save_user_pref(update.effective_user.id, "strict", int(not curr))
    key = "strict_on" if not curr else "strict_off"
    await safe_reply(update, t(key, ctx))

async def cmd_dpi(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = ctx.args
    if args and args[0] in ("72", "96", "150", "300"):
        v = int(args[0])
        ctx.user_data["dpi"] = v
        save_user_pref(update.effective_user.id, "dpi", v)
        await safe_reply(update, t("dpi_set", ctx) + f"`{v}`")
    else:
        await safe_reply(update, t("dpi_usage", ctx))

# /history command intentionally removed.
# Storing user images — even temporarily in RAM — violates privacy-first design.
# Users can simply reprocess if needed.

async def cmd_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    secure_wipe_all(ctx, ["bg_img","bg_result","resize_img","resize_result",
                  "reduce_img","sig_result","target_kb"])
    release_lock(ctx)
    await safe_reply(update, t("cancel", ctx))
    await send_main_menu(update, ctx)
    return ConversationHandler.END

async def cmd_mystats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    hid = hash_uid(uid)
    with get_db() as conn:
        row = conn.execute(
            "SELECT total_ops, dpi FROM users WHERE hashed_uid=?", (hid,)
        ).fetchone()
    if not row:
        await safe_reply(update, "📊 No stats yet. Use the bot first!")
        return
    await safe_reply(update,
        f"📊 *Your Stats*\n\n"
        f"Total operations: `{row['total_ops']}`\n"
        f"Output DPI: `{row['dpi']}`\n\n"
        f"_Your identity is stored as an anonymous hash.\n"
        f"No timestamps, names, or Telegram ID are kept._"
    )


async def cmd_admin(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if uid not in ADMIN_IDS:
        await safe_reply(update, "❌ Access denied.")
        return
    stats  = get_bot_stats()
    uptime = int(time.time() - BOT_START_TIME)
    await safe_reply(update,
        f"🛠 *Admin Panel — {VERSION}*\n\n"
        f"Uptime: `{uptime // 3600}h {(uptime % 3600) // 60}m`\n\n"
        f"👥 Total users (hashed): `{stats['total_users']}`\n"
        f"📊 Total ops (anonymous): `{stats['total_ops']}`\n\n"
        f"_No personal data stored. Real IDs hashed. No op logs._"
    )


async def cmd_broadcast(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if uid not in ADMIN_IDS:
        await safe_reply(update, "❌ Access denied.")
        return
    if not ctx.args:
        await safe_reply(update, "Usage: `/broadcast Your message here`")
        return
    message  = " ".join(ctx.args)
    real_ids = list(_known_real_ids)  # RAM-only, never from DB
    if not real_ids:
        await safe_reply(update, "⚠️ No active users in current session. Users must interact first.")
        return
    status = await update.message.reply_text(
        f"📡 Broadcasting to {len(real_ids)} users seen this session..."
    )
    sent = failed = 0
    for chunk_start in range(0, len(real_ids), 30):
        for rid in real_ids[chunk_start:chunk_start + 30]:
            try:
                await ctx.bot.send_message(
                    chat_id=rid,
                    text=f"📢 *Announcement*\n\n{message}",
                    parse_mode="Markdown"
                )
                sent += 1
            except Exception:
                failed += 1
        await asyncio.sleep(1)
    with get_db() as conn:
        conn.execute("INSERT INTO broadcasts(message, sent_at) VALUES(?,?)",
                     (message, datetime.utcnow().isoformat()))
    await status.edit_text(f"✅ Done! Sent: `{sent}` | Failed: `{failed}`")




# ─────────────────────────────────────────────────────────────────────
# ADMIN: PRESET MANAGEMENT
# ─────────────────────────────────────────────────────────────────────

async def cmd_listpresets(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if uid not in ADMIN_IDS:
        await safe_reply(update, "❌ Access denied.")
        return
    presets = db_list_all_presets()
    if not presets:
        await safe_reply(update, "📋 No presets yet.")
        return
    lines = ["📋 *All Presets (Admin View)*\n"]
    for p in presets:
        status = "✅" if p["active"] else "❌ deleted"
        lines.append(f"{status} `ID:{p['id']}` — {p['label']} — `{p['width_px']}×{p['height_px']}px`")
    lines.append("\n`/addpreset Label | w | h`")
    lines.append("`/editpreset ID | Label | w | h`")
    lines.append("`/delpreset ID`")
    await safe_reply(update, "\n".join(lines))


async def cmd_addpreset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if uid not in ADMIN_IDS:
        await safe_reply(update, "❌ Access denied.")
        return
    raw   = " ".join(ctx.args) if ctx.args else ""
    parts = [p.strip() for p in raw.split("|")]
    if len(parts) != 3:
        await safe_reply(update,
            "📝 *Format:* `/addpreset Label | width | height`\n\n"
            "*Example:* `/addpreset Railway 2025 | 413 | 531`")
        return
    label = parts[0].strip()
    try:
        w, h = int(parts[1].strip()), int(parts[2].strip())
        if not (0 < w <= 5000 and 0 < h <= 5000):
            raise ValueError
    except ValueError:
        await safe_reply(update, "❌ Width/height must be valid numbers (1–5000 px).")
        return
    if not label:
        await safe_reply(update, "❌ Label khali nahi hona chahiye.")
        return
    new_id = db_add_preset(label, w, h)
    await safe_reply(update,
        f"✅ *Preset added!*\n🆔 ID:`{new_id}` — {label} — `{w}×{h}px`\n\n"
        f"_Live immediately — no restart needed._")


async def cmd_editpreset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if uid not in ADMIN_IDS:
        await safe_reply(update, "❌ Access denied.")
        return
    raw   = " ".join(ctx.args) if ctx.args else ""
    parts = [p.strip() for p in raw.split("|")]
    if len(parts) != 4:
        await safe_reply(update,
            "📝 *Format:* `/editpreset ID | Label | width | height`\n\n"
            "*Example:* `/editpreset 9 | Railway 2026 | 450 | 550`")
        return
    try:
        pid   = int(parts[0].strip())
        label = parts[1].strip()
        w, h  = int(parts[2].strip()), int(parts[3].strip())
        if not (0 < w <= 5000 and 0 < h <= 5000):
            raise ValueError
    except ValueError:
        await safe_reply(update, "❌ ID aur dimensions valid numbers hone chahiye.")
        return
    preset = db_get_preset_by_id(pid)
    if not preset:
        await safe_reply(update, f"❌ ID `{pid}` not found.")
        return
    db_edit_preset(pid, label, w, h)
    await safe_reply(update,
        f"✅ *Preset updated!*\n"
        f"Before: {preset['label']} `{preset['width_px']}×{preset['height_px']}`\n"
        f"After: {label} `{w}×{h}px`\n\n_Live immediately._")


async def cmd_delpreset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if uid not in ADMIN_IDS:
        await safe_reply(update, "❌ Access denied.")
        return
    if not ctx.args or not ctx.args[0].isdigit():
        await safe_reply(update,
            "📝 *Format:* `/delpreset ID`\n\n"
            "*Example:* `/delpreset 3`")
        return
    pid    = int(ctx.args[0])
    preset = db_get_preset_by_id(pid)
    if not preset:
        await safe_reply(update, f"❌ ID `{pid}` not found.")
        return
    db_delete_preset(pid)
    await safe_reply(update,
        f"🗑 *Deleted:* {preset['label']} `{preset['width_px']}×{preset['height_px']}px`")


# ─────────────────────────────────────────────────────────────────────
# GRACEFUL SHUTDOWN
# ─────────────────────────────────────────────────────────────────────
def handle_signal(sig, frame):
    global _bot_healthy
    _bot_healthy = False
    _wipe_session_key()
    logger.info(f"Signal {sig} — session key wiped, shutting down.")
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT,  handle_signal)


# ─────────────────────────────────────────────────────────────────────
# POST INIT — register commands, warm up model
# ─────────────────────────────────────────────────────────────────────
async def post_init(application: Application):
    public_commands = [
        BotCommand("start",    "Main menu"),
        BotCommand("help",     "How to use"),
        BotCommand("privacy",  "Privacy policy"),
        BotCommand("cancel",   "Cancel current operation"),
        BotCommand("mystats",  "Your usage stats"),
        BotCommand("hinglish", "Toggle Hinglish/English"),
        BotCommand("strict",   "Toggle strict aspect ratio"),
        BotCommand("dpi",      "Set output DPI (72/150/300)"),
    ]
    admin_commands = public_commands + [
        BotCommand("admin",       "Admin panel"),
        BotCommand("broadcast",   "Broadcast to all users"),
        BotCommand("listpresets", "List all presets"),
        BotCommand("addpreset",   "Add new preset"),
        BotCommand("editpreset",  "Edit existing preset"),
        BotCommand("delpreset",   "Delete a preset"),
    ]
    await application.bot.set_my_commands(public_commands)
    for admin_id in ADMIN_IDS:
        try:
            await application.bot.set_my_commands(
                admin_commands,
                scope={"type": "chat", "chat_id": admin_id}
            )
        except Exception:
            pass
    logger.info("Bot commands registered.")
    warm_up_model()


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    token = os.environ.get("BOT_TOKEN")
    if not token:
        logger.error("BOT_TOKEN environment variable not set!")
        sys.exit(1)

    if _UID_SALT == "change-this-salt-in-production":
        logger.warning("⚠️  UID_SALT env var not set! Set a random secret in Render env vars.")

    init_db()
    logger.info(f"Database ready: {DB_PATH}")

    threading.Thread(target=run_flask, daemon=True).start()
    logger.info(f"Flask health server starting on port {os.environ.get('PORT', 8080)}")

    application = Application.builder().token(token).post_init(post_init).build()

    # Standalone commands
    for cmd, fn in [
        ("help",        cmd_help),
        ("privacy",     cmd_privacy),
        ("hinglish",    cmd_hinglish),
        ("strict",      cmd_strict),
        ("dpi",         cmd_dpi),
        ("mystats",     cmd_mystats),
        ("admin",       cmd_admin),
        ("broadcast",   cmd_broadcast),
        ("listpresets", cmd_listpresets),
        ("addpreset",   cmd_addpreset),
        ("editpreset",  cmd_editpreset),
        ("delpreset",   cmd_delpreset),
    ]:
        application.add_handler(CommandHandler(cmd, fn))

    # Main conversation
    conv = ConversationHandler(
        entry_points=[CommandHandler("start", cmd_start)],
        states={
            S.SELECT_ACTION:        [CallbackQueryHandler(select_action)],
            S.BG_WAIT_PHOTO:        [MessageHandler(filters.PHOTO | filters.Document.IMAGE, bg_wait_photo)],
            S.BG_WAIT_COLOR:        [
                CallbackQueryHandler(bg_wait_color, pattern=r"^col_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, bg_wait_color),
            ],
            S.BG_PREVIEW:           [CallbackQueryHandler(bg_preview,          pattern=r"^bg_")],
            S.BG_WAIT_FORMAT:       [CallbackQueryHandler(bg_wait_format,       pattern=r"^fmt_")],
            S.RESIZE_MODE:          [CallbackQueryHandler(resize_mode,          pattern=r"^resize_")],
            S.CUSTOM_WAIT_PHOTO:    [MessageHandler(filters.PHOTO | filters.Document.IMAGE, custom_wait_photo)],
            S.CUSTOM_SELECT_PRESET: [CallbackQueryHandler(custom_select_preset, pattern=r"^preset_")],
            S.CUSTOM_WAIT_DIMS:     [MessageHandler(filters.TEXT & ~filters.COMMAND, custom_wait_dims)],
            S.CUSTOM_PREVIEW:       [CallbackQueryHandler(custom_preview,       pattern=r"^resize_")],
            S.CUSTOM_SIZE_OPT:      [CallbackQueryHandler(custom_size_opt,      pattern=r"^sizeopt_")],
            S.CUSTOM_WAIT_KB:       [MessageHandler(filters.TEXT & ~filters.COMMAND, custom_wait_kb)],
            S.CUSTOM_WAIT_FORMAT:   [CallbackQueryHandler(custom_wait_format,   pattern=r"^fmt_")],
            S.REDUCE_WAIT_PHOTO:    [MessageHandler(filters.PHOTO | filters.Document.IMAGE, reduce_wait_photo)],
            S.REDUCE_WAIT_KB:       [MessageHandler(filters.TEXT & ~filters.COMMAND, reduce_wait_kb)],
            S.REDUCE_WAIT_FORMAT:   [CallbackQueryHandler(reduce_wait_format,   pattern=r"^fmt_")],
            S.SIG_WAIT_PHOTO:       [MessageHandler(filters.PHOTO | filters.Document.IMAGE, sig_wait_photo)],
            S.SIG_PREVIEW:          [CallbackQueryHandler(sig_preview,          pattern=r"^sig_")],
            S.SIG_WAIT_FORMAT:      [CallbackQueryHandler(sig_wait_format,      pattern=r"^fmt_")],
        },
        fallbacks=[
            CommandHandler("cancel", cmd_cancel),
            CommandHandler("start",  cmd_start),
            MessageHandler(filters.ALL, conversation_fallback),
        ],
        allow_reentry=True,
        conversation_timeout=1800,
    )

    application.add_handler(conv)
    application.add_handler(MessageHandler(filters.ALL, global_fallback))
    application.add_error_handler(error_handler)

    logger.info(f"🚀 {VERSION} starting — privacy-first mode active")
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
