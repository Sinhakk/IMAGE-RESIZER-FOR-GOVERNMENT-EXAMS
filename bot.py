#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║      IMAGE UTILITY BOT  —  PRODUCTION  v6.5  (fully audited)     ║
║      Government Exam Photo Helper                                ║
║                                                                  ║
║  CORE:                                                           ║
║   • AI BG Change v6.2 (trimap matting + color decontamination)   ║
║   • Face-Aware Passport Crop v6.5 (crop-in-source: zero giant    ║
║     intermediates, exact govt-spec positioning at ANY scale)     ║
║   • Print Sheet (8 photos, 4×6", sanity fallback)                ║
║   • Signature v2.1 (hysteresis + auto-downscale + empty-detect)  ║
║   • Binary-Search Compress (±2%)                                 ║
║   • 🎯 EXACT SIZE MATCH — same resolution, KB up OR down         ║
║   • Auto-Enhance (gentle for faces) + Quality Report             ║
║   • HEIC support • /history encrypted resend                     ║
║                                                                  ║
║  v6.5 AUDIT FIXES:                                               ║
║   • resize_mode double-answer BadRequest — FIXED                 ║
║   • First-visit user DB row now ALWAYS created (stats correct)   ║
║   • passport_crop memory bomb eliminated (no 2x-cap distortion)  ║
║   • Image decode moved off event loop (no multi-user freezes)    ║
║   • Signature auto-downscale >2000px (10× faster on big scans)   ║
║   • Idle-expiry now deletes ghost prompt + notifies              ║
║   • Empty-signature warning • dead code removed                  ║
║                                                                  ║
║  PRIVACY: AES-256-GCM RAM • hashed IDs • nothing on disk         ║
║  INFRA: Render.com ready (waitress health + polling)             ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────
import os, io, re, cv2, sqlite3, logging, time, asyncio, gc
import sys, threading, hashlib
from enum import Enum
from uuid import uuid4
from functools import lru_cache
from collections import defaultdict
from threading import Lock
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any
from contextlib import contextmanager
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import numpy as np
from PIL import Image, ImageColor, ImageOps, ImageDraw

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_OK = True
except ImportError:
    HEIC_OK = False

try:
    import mediapipe as mp
    MP_OK = True
except ImportError:
    MP_OK = False

try:
    from waitress import serve as _waitress_serve
    WAITRESS_OK = True
except ImportError:
    WAITRESS_OK = False

from flask import Flask, jsonify
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ConversationHandler, ContextTypes, filters,
)
from telegram.error import BadRequest, TimedOut, NetworkError, RetryAfter

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
VERSION          = "v6.5-production"
DPI_DEFAULT      = 300
MAX_QUALITY      = 95
MIN_QUALITY      = 10
PREVIEW_MAX_SIZE = (512, 512)
MAX_IMAGE_DIM    = 4096
MIN_DIM          = 16
BG_MAX_DIM       = 2048           # BG-change memory guard (512MB tier)
SIG_MAX_DIM      = 2000           # signature scan cap — bilateral filter speed
NOISE_MAX_PIXELS = 4_000_000      # noise-injection cap — RAM safety
PROCESSING_TIMEOUT = 90
RATE_LIMIT_REQ   = 8
RATE_LIMIT_SECS  = 60
MAX_FILE_SIZE_MB = 20
SESSION_IDLE_SECS = 1800
ALLOWED_DPIS     = {72, 96, 150, 300, 600}
ADMIN_IDS_RAW    = os.environ.get("ADMIN_IDS", "")
ADMIN_IDS        = {int(x.strip()) for x in ADMIN_IDS_RAW.split(",") if x.strip().isdigit()}
DB_PATH          = os.environ.get("DB_PATH", "/tmp/imagebot.db")
BOT_START_TIME   = time.time()

# ─────────────────────────────────────────────────────────────────────
# PRIVACY CORE — AES-256-GCM
# ─────────────────────────────────────────────────────────────────────
_SESSION_KEY: bytes = AESGCM.generate_key(bit_length=256)
_AESGCM               = AESGCM(_SESSION_KEY)
_UID_SALT             = os.environ.get("UID_SALT", "change-this-salt-in-production")

import atexit as _atexit

def hash_uid(telegram_id: int) -> str:
    return hashlib.sha256(f"{_UID_SALT}:{telegram_id}".encode("utf-8")).hexdigest()

class SecureBuffer:
    """AES-256-GCM holder. bytearray = real zero-wipe."""
    __slots__ = ("_ct", "_nonce", "_wiped")

    def __init__(self, plaintext: bytes):
        self._nonce = bytearray(os.urandom(12))
        self._ct    = bytearray(_AESGCM.encrypt(bytes(self._nonce), bytes(plaintext), None))
        self._wiped = False

    def decrypt(self) -> bytes:
        if self._wiped:
            raise RuntimeError("SecureBuffer already wiped")
        return _AESGCM.decrypt(bytes(self._nonce), bytes(self._ct), None)

    def wipe(self):
        if not self._wiped:
            self._ct[:]    = b"\x00" * len(self._ct)
            self._nonce[:] = b"\x00" * 12
            self._wiped = True
            gc.collect()

    @contextmanager
    def open(self):
        plaintext = None
        try:
            plaintext = self.decrypt()
            yield plaintext
        finally:
            del plaintext
            self.wipe()

    def __del__(self):
        try:
            self.wipe()
        except Exception:
            pass

def secure_store(ctx, key: str, img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    buf.close()
    ctx.user_data[key] = SecureBuffer(data)
    del data
    gc.collect()

def secure_load(ctx, key: str) -> Optional[Image.Image]:
    sbuf = ctx.user_data.pop(key, None)
    if not isinstance(sbuf, SecureBuffer):
        return None
    try:
        with sbuf.open() as data:
            img = Image.open(io.BytesIO(data)).copy()
        gc.collect()
        return img
    except Exception:
        return None

def secure_peek(ctx, key: str) -> Optional[bytes]:
    sbuf = ctx.user_data.get(key)
    if not isinstance(sbuf, SecureBuffer):
        return None
    try:
        return sbuf.decrypt()
    except RuntimeError:
        return None

def secure_wipe_all(ctx, keys: list):
    for key in keys:
        val = ctx.user_data.pop(key, None)
        if isinstance(val, SecureBuffer):
            val.wipe()
    gc.collect()

# Async wrappers — PNG encode/decode HEAVY, hamesha thread mein
async def a_secure_store(ctx, key: str, img: Image.Image):
    await asyncio.to_thread(secure_store, ctx, key, img)

async def a_secure_load(ctx, key: str) -> Optional[Image.Image]:
    return await asyncio.to_thread(secure_load, ctx, key)

# Session keys — har op ke baad wipe. hist_buf YAHAAN NAHI (history survives).
_IMAGE_KEYS = [
    "bg_img", "bg_result",
    "resize_img", "resize_result",
    "reduce_img", "sig_result",
    "size_img",
]

def _wipe_session_key():
    global _SESSION_KEY
    try:
        _SESSION_KEY = bytes(len(_SESSION_KEY))
        del _SESSION_KEY
    except Exception:
        pass

_atexit.register(_wipe_session_key)

# ─────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("ImageBot")

if _UID_SALT == "change-this-salt-in-production":
    logger.warning("⚠️  UID_SALT env var not set — using insecure default!")

# ─────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────
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

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                hashed_uid  TEXT PRIMARY KEY,
                total_ops   INTEGER DEFAULT 0,
                hinglish    INTEGER DEFAULT 0,
                strict      INTEGER DEFAULT 1,
                dpi         INTEGER DEFAULT 300
            );
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
        if conn.execute("SELECT COUNT(*) FROM presets").fetchone()[0] == 0:
            _seed_presets(conn)

def _seed_presets(conn):
    defaults = [
        ("passport_india",  "🪪 Passport India (3.5×4.5cm)",  413, 531, 1),
        ("passport_us",     "🇺🇸 US Passport (2×2 inch)",     600, 600, 2),
        ("stamp_size",      "📮 Stamp Size (2.5×3cm)",        295, 354, 3),
        ("aadhaar",         "🆔 Aadhaar (3.5×4.5cm)",         413, 531, 4),
        ("pan_card",        "💳 PAN Card (3.5×4.5cm)",        413, 531, 5),
        ("driving",         "🚗 Driving License (3.5×4.5cm)", 413, 531, 6),
        ("upsc",            "📋 UPSC (4.5×4.5cm)",            531, 531, 7),
        ("ssc",             "📋 SSC (3.5×4.5cm)",             413, 531, 8),
        ("railway",         "🚂 Railway (3.5×4.5cm)",         413, 531, 9),
        ("neet",            "⚕️ NEET/JEE (4.5×4.5cm)",        531, 531, 10),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO presets(slug,label,width_px,height_px,sort_order) VALUES(?,?,?,?,?)",
        defaults)

def db_get_presets() -> list:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id,slug,label,width_px,height_px FROM presets WHERE active=1 ORDER BY sort_order,id"
        ).fetchall()
    return [dict(r) for r in rows]

def db_get_preset_by_id(pid: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT id,slug,label,width_px,height_px FROM presets WHERE id=? AND active=1",
            (pid,)).fetchone()
    return dict(row) if row else None

def db_add_preset(label: str, w: int, h: int) -> int:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower().strip())[:30] or "preset"
    with get_db() as conn:
        base, suffix = slug, 1
        while conn.execute("SELECT 1 FROM presets WHERE slug=?", (slug,)).fetchone():
            slug = f"{base}_{suffix}"; suffix += 1
        max_o = conn.execute("SELECT MAX(sort_order) FROM presets").fetchone()[0] or 0
        cur = conn.execute(
            "INSERT INTO presets(slug,label,width_px,height_px,sort_order) VALUES(?,?,?,?,?)",
            (slug, label, w, h, max_o + 1))
        return cur.lastrowid

def db_edit_preset(pid: int, label: str, w: int, h: int):
    with get_db() as conn:
        conn.execute("UPDATE presets SET label=?,width_px=?,height_px=? WHERE id=?",
                     (label, w, h, pid))

def db_delete_preset(pid: int):
    with get_db() as conn:
        conn.execute("UPDATE presets SET active=0 WHERE id=?", (pid,))

def db_list_all_presets() -> list:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id,label,width_px,height_px,active,sort_order FROM presets ORDER BY sort_order,id"
        ).fetchall()
    return [dict(r) for r in rows]

def upsert_user(user_id: int):
    with get_db() as conn:
        conn.execute("INSERT OR IGNORE INTO users(hashed_uid) VALUES(?)", (hash_uid(user_id),))

def get_user_prefs(user_id: int) -> Dict[str, Any]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT hinglish,strict,dpi FROM users WHERE hashed_uid=?", (hash_uid(user_id),)
        ).fetchone()
    if row:
        return {"hinglish": bool(row["hinglish"]), "strict": bool(row["strict"]), "dpi": row["dpi"]}
    return {"hinglish": False, "strict": True, "dpi": DPI_DEFAULT}

def save_user_pref(user_id: int, key: str, value):
    if key not in {"hinglish", "strict", "dpi"}:
        return
    with get_db() as conn:
        conn.execute(f"UPDATE users SET {key}=? WHERE hashed_uid=?", (value, hash_uid(user_id)))

def bump_op_count(user_id: int):
    with get_db() as conn:
        conn.execute("UPDATE users SET total_ops=total_ops+1 WHERE hashed_uid=?",
                     (hash_uid(user_id),))

def get_bot_stats() -> Dict[str, Any]:
    with get_db() as conn:
        users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        ops   = conn.execute("SELECT SUM(total_ops) FROM users").fetchone()[0] or 0
    return {"total_users": users, "total_ops": ops}

_known_real_ids: set = set()

# ─────────────────────────────────────────────────────────────────────
# MEDIAPIPE — thread-safe singletons
# ─────────────────────────────────────────────────────────────────────
_mp_model = None
_mp_face  = None
_mp_init_lock = Lock()
_mp_seg_lock  = Lock()
_mp_face_lock = Lock()

def get_segmentation_model():
    global _mp_model
    with _mp_init_lock:
        if _mp_model is None:
            if not MP_OK:
                raise RuntimeError("mediapipe not installed")
            logger.info("Warming up MediaPipe segmentation...")
            _mp_model = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            _mp_model.process(np.zeros((100, 100, 3), dtype=np.uint8))
            logger.info("Segmentation model ready.")
    return _mp_model

def run_segmentation(rgb_arr: np.ndarray):
    model = get_segmentation_model()
    with _mp_seg_lock:
        return model.process(rgb_arr)

def get_face_detector():
    global _mp_face
    with _mp_init_lock:
        if _mp_face is None and MP_OK:
            logger.info("Warming up MediaPipe face detection...")
            _mp_face = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5)
    return _mp_face

def run_face_detect(rgb_arr: np.ndarray):
    det = get_face_detector()
    if det is None:
        return None
    with _mp_face_lock:
        return det.process(rgb_arr)

def warm_up_model():
    if MP_OK:
        threading.Thread(target=get_segmentation_model, daemon=True).start()
        threading.Thread(target=get_face_detector, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────
# FLASK / WAITRESS HEALTH
# ─────────────────────────────────────────────────────────────────────
flask_app = Flask(__name__)
_bot_healthy = True

@flask_app.route("/")
def home():
    return jsonify({
        "status": "running" if _bot_healthy else "degraded",
        "bot": VERSION,
        "uptime_seconds": int(time.time() - BOT_START_TIME),
    })

@flask_app.route("/health")
def health():
    return (jsonify({"status": "healthy"}), 200) if _bot_healthy \
        else (jsonify({"status": "degraded"}), 503)

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    if WAITRESS_OK:
        _waitress_serve(flask_app, host="0.0.0.0", port=port, threads=4)
    else:
        flask_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

# ─────────────────────────────────────────────────────────────────────
# LANGUAGE
# ─────────────────────────────────────────────────────────────────────
STRINGS: Dict[str, Dict[str, str]] = {
    "hi": {
        "main_menu":           "🏠 *Main Menu* — kya karna hai?",
        "bg_change":           "🖼 Background Change",
        "resize":              "📐 Resize / Compress",
        "signature":           "✍️ Signature Extract",
        "size_match":          "🎯 Exact Size Match",
        "print_sheet":         "🖨 Print Sheet (8 photos)",
        "send_photo":          "📸 Photo bhejo (JPEG/PNG/HEIC, max 20MB):",
        "processing":          "⏳ Processing ho raha hai... thoda intezaar karo.",
        "preview":             "👀 *Preview ready!* Theek lag raha hai?",
        "looks_ok":            "✅ Theek hai — aage badho",
        "retry":               "🔁 Dobara try karo",
        "format_choose":       "📁 Format select karo:",
        "dimensions":          "📐 Dimensions batao (koi bhi format):\n• `300x400` ya `300x400px`\n• `3.5x4.5cm` ya `3.5cm x 4.5cm`\n• `35x45mm`\n• `2x2in`",
        "color_choose":        "🎨 Background color select karo:",
        "custom_color_prompt": "🖊 Color type karo (naam ya hex):\nExamples: `white`, `blue`, `#E8F4FD`",
        "enter_kb":            "📦 Target size batao:\nExample: `100` (KB), `1.5mb`",
        "size_option":         "📦 Size kaise set karna hai?",
        "size_by_kb":          "📦 KB/MB target set karo",
        "size_by_dims":        "🖼 Format ke saath direct save karo",
        "reduce_send_photo":   "📸 Jo photo compress karni hai wo bhejo:",
        "select_preset":       "📋 Photo type select karo (ya Custom Dimensions):",
        "blur_warn":           "⚠️ Photo blur hai — phir bhi process kar raha hun.",
        "face_small":          "⚠️ Face chhota hai — closer photo better rahegi.",
        "face_offcenter":      "⚠️ Face center mein nahi — framing sudhaaro.",
        "sig_bg_warn":         "⚠️ Background safed nahi lag raha. Safed paper best rahega.",
        "sig_empty_warn":      "⚠️ Koi signature detect NAHI hua — photo clear nahi hai ya ink bahut light hai. 🔁 se dobara try karo.",
        "bg_warning":          "ℹ️ AI result — preview dhyan se check karo.",
        "reminder":            "⚠️ Upload se *pehle* result verify karo.",
        "rate_limit":          "⏳ Max 8 ops/minute. Thodi der baad try karo.",
        "timeout_err":         "⏱ Timeout. Chhoti photo try karo.",
        "file_too_large":      "❌ File 20MB se badi hai.",
        "invalid_file":        "❌ Ye image nahi hai. JPEG/PNG/HEIC bhejo.",
        "no_photo":            "❌ Photo nahi mili. Dobara bhejo.",
        "bg_done":             "✅ *Background change ho gaya!*",
        "resize_done":         "✅ *Resize ho gaya!*",
        "compress_done":       "✅ *Compress ho gaya!*",
        "sig_done":            "✅ *Signature extract ho gaya!*",
        "cancel":              "✋ Cancel ho gaya.",
        "error":               "❌ Kuch gadbad hui. /start karo.",
        "unexpected":          "🤔 Abhi is step pe ye kaam nahi hota. Upar diye steps follow karo.",
        "processing_lock":     "⚙️ Ek operation chal raha hai. Khatam hone do ya /cancel karo.",
        "history":             "📤 Last processed image:",
        "no_history":          "📭 Koi previous image nahi.",
        "strict_on":           "✅ Strict ON — white padding, *no distortion*.",
        "strict_off":          "✂️ Crop mode — center-crop, edges cut ho sakte hain.",
        "dpi_set":             "✅ DPI set: ",
        "dpi_usage":           "Usage: `/dpi 72`, `/dpi 96`, `/dpi 150`, `/dpi 300`, `/dpi 600`",
        "expired":             "⌛ 30 min idle — session clear. /start dabao.",
        "ai_unavailable":      "⚠️ AI model load nahi hua. Admin ko batao.",
        "lang_hi":             "✅ Hinglish ON! 🇮🇳",
        "lang_en":             "✅ English ON! 🇬🇧",
        "q_ok":                "✨ Quality check: Sharp ✓ Lighting ✓",
        "q_dark":              "🌑 Dark thi — auto-brightness laga di.",
        "q_bright":            "☀️ Overexposed thi — auto-fix laga di.",
        "q_flat":              "🌒 Contrast low tha — auto-enhance laga di.",
        "size_send_photo":     "📸 Wo photo bhejo jiska SIZE match karna hai:\n\n🔒 Resolution 100% SAME rahegi — koi resize NAHI.",
        "size_enter_kb":       "📦 *EXACT* target size batao (KB):\nExample: `100` ya `150kb` ya `0.5mb`\n\n💡 Size KAM ya ZYADA dono ho sakta hai — resolution same rahegi.",
        "size_fmt_choose":     "📁 Format chuno (JPEG recommended — exact size hit hota hai):",
        "size_done":           "🎯 *Size matched! Resolution SAME rakha.*",
        "size_min_warn":       "⚠️ Itna chhota size is resolution pe possible nahi — best possible de diya. Aur chhota chahiye to /resize use karo.",
        "size_max_warn":       "⚠️ Itna bada size possible nahi — max reached, best de diya.",
        "size_q_warn":         "ℹ️ Target ke liye compression thodi zyada lagi (q{q}).",
        "size_info":           "📐 Resolution: `{w}×{h}` (UNCHANGED)\n📦 Size: `{size}` (target: `{target}`)\n🎚 Quality: `{q}`",
        "sheet_send_photo":    "📸 Photo bhejo — 4×6\" sheet pe *8 passport photos* (cut lines ke saath) ban jayegi:",
        "sheet_done":          "🖨 *Print sheet ready!*\n📄 4×6\" | 8 photos | Passport 35×45mm\n\n💡 Print shop ko JPEG do ya PDF print karo.",
        "privacy_notice": (
            "🔒 *Privacy Policy*\n\n"
            "• Photos RAM mein AES-256 encrypted, process hote hi wiped\n"
            "• Kuch bhi disk pe save NAHI hota\n"
            "• Telegram ID sirf irreversible hash ke roop mein\n"
            "• Naam/username/timestamp — kuch nahi\n"
            "• Sirf anonymous op counter\n\n"
            "_Privacy pehli priority._"
        ),
        "help_text": (
            "📖 *Bot Guide*\n\n"
            "🖼 *Background Change* — AI se koi bhi color, clean edges\n\n"
            "📐 *Resize* — Presets (Passport/UPSC/NEET...), custom units, compress\n"
            "   • Face auto-position: govt-spec 70% face height\n\n"
            "🎯 *Exact Size Match* — resolution SAME, size KAM ya ZYADA exact KB\n\n"
            "🖨 *Print Sheet* — 8 passport photos 4×6\" pe (print-ready)\n\n"
            "✍️ *Signature* — transparent PNG / white JPEG / PDF\n\n"
            "⚙️ *Commands*\n"
            "/start /cancel /history /hinglish /strict /dpi /privacy /help\n\n"
            "💡 *Tips*\n"
            "• Clear, well-lit photo = best result\n"
            "• iPhone HEIC supported\n"
            "• Signature: dark ink, white paper"
        ),
    },
    "en": {
        "main_menu":           "🏠 *Main Menu* — What would you like to do?",
        "bg_change":           "🖼 Background Change",
        "resize":              "📐 Resize / Compress",
        "signature":           "✍️ Signature Extract",
        "size_match":          "🎯 Exact Size Match",
        "print_sheet":         "🖨 Print Sheet (8 photos)",
        "send_photo":          "📸 Send your photo (JPEG/PNG/HEIC, max 20MB):",
        "processing":          "⏳ Processing... please wait.",
        "preview":             "👀 *Preview ready!* Does it look OK?",
        "looks_ok":            "✅ Looks good — proceed",
        "retry":               "🔁 Try again",
        "format_choose":       "📁 Choose output format:",
        "dimensions":          "📐 Enter dimensions (any format):\n• `300x400` or `300x400px`\n• `3.5x4.5cm` or `3.5cm x 4.5cm`\n• `35x45mm`\n• `2x2in`",
        "color_choose":        "🎨 Choose background color:",
        "custom_color_prompt": "🖊 Type a color name or hex:\nExamples: `white`, `blue`, `#E8F4FD`",
        "enter_kb":            "📦 Enter target size:\nExample: `100` (KB), `1.5mb`",
        "size_option":         "📦 How do you want to set the size?",
        "size_by_kb":          "📦 Set KB/MB target",
        "size_by_dims":        "🖼 Save directly with format",
        "reduce_send_photo":   "📸 Send the photo to compress:",
        "select_preset":       "📋 Choose photo type (or Custom Dimensions):",
        "blur_warn":           "⚠️ Photo is blurry — processing anyway.",
        "face_small":          "⚠️ Face is small — a closer photo works better.",
        "face_offcenter":      "⚠️ Face is off-center — adjust framing.",
        "sig_bg_warn":         "⚠️ Background doesn't look white. Plain white paper is best.",
        "sig_empty_warn":      "⚠️ No signature detected — the photo may be unclear or the ink too light. Try again with 🔁.",
        "bg_warning":          "ℹ️ AI result — please review the preview carefully.",
        "reminder":            "⚠️ Always verify before uploading.",
        "rate_limit":          "⏳ Max 8 ops/minute. Please wait.",
        "timeout_err":         "⏱ Timed out. Try a smaller image.",
        "file_too_large":      "❌ File exceeds 20MB.",
        "invalid_file":        "❌ Not an image. Send JPEG/PNG/HEIC.",
        "no_photo":            "❌ No photo found. Try again.",
        "bg_done":             "✅ *Background changed!*",
        "resize_done":         "✅ *Image resized!*",
        "compress_done":       "✅ *File compressed!*",
        "sig_done":            "✅ *Signature extracted!*",
        "cancel":              "✋ Cancelled.",
        "error":               "❌ Something went wrong. /start again.",
        "unexpected":          "🤔 That doesn't work at this step. Follow the steps above.",
        "processing_lock":     "⚙️ An operation is running. Wait for it or /cancel.",
        "history":             "📤 Last processed image:",
        "no_history":          "📭 No previous image found.",
        "strict_on":           "✅ Strict ON — white padding, *no distortion*.",
        "strict_off":          "✂️ Crop mode — center-crop, edges may be cut.",
        "dpi_set":             "✅ DPI set to: ",
        "dpi_usage":           "Usage: `/dpi 72`, `/dpi 96`, `/dpi 150`, `/dpi 300`, `/dpi 600`",
        "expired":             "⌛ 30 min idle — session cleared. Press /start.",
        "ai_unavailable":      "⚠️ AI model unavailable. Contact admin.",
        "lang_hi":             "✅ Hinglish ON! 🇮🇳",
        "lang_en":             "✅ English ON! 🇬🇧",
        "q_ok":                "✨ Quality check: Sharp ✓ Lighting ✓",
        "q_dark":              "🌑 Photo was dark — auto-brightness applied.",
        "q_bright":            "☀️ Overexposed — auto-fixed.",
        "q_flat":              "🌒 Low contrast — auto-enhanced.",
        "size_send_photo":     "📸 Send the photo whose SIZE you want to match:\n\n🔒 Resolution stays 100% SAME — no resizing.",
        "size_enter_kb":       "📦 Enter the *EXACT* target size (KB):\nExample: `100` or `150kb` or `0.5mb`\n\n💡 Size can go DOWN or UP — resolution unchanged.",
        "size_fmt_choose":     "📁 Choose format (JPEG recommended — hits exact size):",
        "size_done":           "🎯 *Size matched! Resolution kept SAME.*",
        "size_min_warn":       "⚠️ That size isn't reachable at this resolution — delivered the best possible. Use /resize if smaller dimensions are OK.",
        "size_max_warn":       "⚠️ That size isn't reachable — delivered maximum possible.",
        "size_q_warn":         "ℹ️ Extra compression applied to hit target (q{q}).",
        "size_info":           "📐 Resolution: `{w}×{h}` (UNCHANGED)\n📦 Size: `{size}` (target: `{target}`)\n🎚 Quality: `{q}`",
        "sheet_send_photo":    "📸 Send a photo — I'll make a 4×6\" sheet with *8 passport photos* (with cut lines):",
        "sheet_done":          "🖨 *Print sheet ready!*\n📄 4×6\" | 8 photos | Passport 35×45mm\n\n💡 Give the JPEG to any print shop, or print the PDF.",
        "privacy_notice": (
            "🔒 *Privacy Policy*\n\n"
            "• Photos AES-256 encrypted in RAM, wiped after processing\n"
            "• Nothing written to disk\n"
            "• Telegram ID stored only as an irreversible hash\n"
            "• No name, username, or timestamps\n"
            "• Only an anonymous op counter\n\n"
            "_Privacy comes first._"
        ),
        "help_text": (
            "📖 *Bot Guide*\n\n"
            "🖼 *Background Change* — any color, clean AI edges\n\n"
            "📐 *Resize* — Presets (Passport/UPSC/NEET...), custom units, compress\n"
            "   • Face auto-position: govt-spec 70% face height\n\n"
            "🎯 *Exact Size Match* — same resolution, exact KB (down OR up)\n\n"
            "🖨 *Print Sheet* — 8 passport photos on 4×6\" (print-ready)\n\n"
            "✍️ *Signature* — transparent PNG / white JPEG / PDF\n\n"
            "⚙️ *Commands*\n"
            "/start /cancel /history /hinglish /strict /dpi /privacy /help\n\n"
            "💡 *Tips*\n"
            "• Clear, well-lit photos work best\n"
            "• iPhone HEIC supported\n"
            "• Signature: dark ink, white paper"
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
    BG_WAIT_PHOTO        = 1
    BG_WAIT_COLOR        = 2
    BG_PREVIEW           = 3
    BG_WAIT_FORMAT       = 4
    RESIZE_MODE          = 5
    CUSTOM_WAIT_PHOTO    = 6
    CUSTOM_SELECT_PRESET = 7
    CUSTOM_WAIT_DIMS     = 8
    CUSTOM_PREVIEW       = 9
    CUSTOM_SIZE_OPT      = 10
    CUSTOM_WAIT_KB       = 11
    CUSTOM_WAIT_FORMAT   = 12
    REDUCE_WAIT_PHOTO    = 13
    REDUCE_WAIT_KB       = 14
    REDUCE_WAIT_FORMAT   = 15
    SIG_WAIT_PHOTO       = 16
    SIG_PREVIEW          = 17
    SIG_WAIT_FORMAT      = 18
    SHEET_WAIT_PHOTO     = 19
    SIZE_WAIT_PHOTO      = 20
    SIZE_WAIT_KB         = 21
    SIZE_WAIT_FORMAT     = 22

MENU_ACTION_PATTERN = r"^(bg_change|resize|signature|size_match|print_sheet)$"

# ─────────────────────────────────────────────────────────────────────
# RATE LIMITING + LOCK + TOKEN CLEANUP
# ─────────────────────────────────────────────────────────────────────
_rate_data: dict = defaultdict(list)

def check_rate_limit(user_id: int) -> bool:
    now = time.time()
    dq = _rate_data[user_id]
    while dq and now - dq[0] >= RATE_LIMIT_SECS:
        dq.pop(0)
    if len(_rate_data) > 1000:
        for k in [k for k, v in _rate_data.items()
                  if not v or now - v[-1] >= RATE_LIMIT_SECS][:600]:
            _rate_data.pop(k, None)
    if len(dq) >= RATE_LIMIT_REQ:
        return False
    dq.append(now)
    return True

def acquire_lock(ctx) -> bool:
    if ctx.user_data.get("_lock"):
        return False
    ctx.user_data["_lock"] = True
    return True

def release_lock(ctx):
    ctx.user_data.pop("_lock", None)

def new_op_token(ctx) -> str:
    tok = uuid4().hex
    ctx.user_data["_op"] = tok
    return tok

def cleanup_session(ctx):
    """Op data wipe — history (hist_buf) preserve hoti hai."""
    secure_wipe_all(ctx, _IMAGE_KEYS + ["target_kb", "_resize_mode"])
    ctx.user_data["_op"] = None
    release_lock(ctx)

async def schedule_cleanup(ctx, token: str, delay: int = SESSION_IDLE_SECS):
    await asyncio.sleep(delay)
    if ctx.user_data.get("_op") != token:
        return
    cleanup_session(ctx)
    # v6.5: ghost prompt bhi delete karo — dead buttons kabhi na bachein
    chat_id = ctx.user_data.get("_chat_id")
    mid = ctx.user_data.pop("svc_msg_id", None)
    if chat_id and mid:
        try:
            await ctx.bot.delete_message(chat_id, mid)
        except Exception:
            pass
    if chat_id:
        try:
            await ctx.bot.send_message(chat_id, t("expired", ctx))
        except Exception:
            pass

def _start_op(update: Update, ctx) -> str:
    token = new_op_token(ctx)
    ctx.user_data["_chat_id"] = update.effective_chat.id
    asyncio.create_task(schedule_cleanup(ctx, token))
    return token

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
    return img.convert("RGB") if img.mode != "RGB" else img

def flatten_on_white(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA", "PA"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return ensure_rgb(img)

def downscale_if_needed(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIM:
        s = MAX_IMAGE_DIM / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)
    return img

def _decode_image(data: bytes) -> Image.Image:
    """v6.5: CPU-heavy decode/orient/downscale — thread se call hota hai."""
    img = Image.open(io.BytesIO(data))
    img = fix_orientation(img)
    img = ensure_rgb(img)
    img = downscale_if_needed(img)
    return img

@lru_cache(maxsize=1)
def _haar_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def analyze_face(img: Image.Image) -> Optional[str]:
    try:
        gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        faces = _haar_cascade().detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
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

def analyze_quality(img: Image.Image) -> Dict[str, Any]:
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    blur  = cv2.Laplacian(gray, cv2.CV_64F).var()
    bright = float(gray.mean())
    contrast = float(gray.std())
    issues = []
    if blur < 60:       issues.append("blur")
    if bright < 90:     issues.append("dark")
    if bright > 225:    issues.append("bright")
    if contrast < 35:   issues.append("flat")
    return {"blur": blur, "brightness": bright, "contrast": contrast, "issues": issues}

def quality_feedback(img: Image.Image, ctx) -> str:
    q = analyze_quality(img)
    parts = []
    if "blur" in q["issues"]:
        parts.append(t("blur_warn", ctx))
    if "dark" in q["issues"]:
        parts.append(t("q_dark", ctx))
    if "bright" in q["issues"]:
        parts.append(t("q_bright", ctx))
    if "flat" in q["issues"]:
        parts.append(t("q_flat", ctx))
    if not parts:
        return t("q_ok", ctx)
    return "\n".join(parts)

def _preflight(img: Image.Image, ctx, gentle: bool):
    """Thread-runner: quality report + face warn + enhance — ek saath."""
    fb = quality_feedback(img, ctx)
    fw = analyze_face(img)
    return fb, fw, auto_enhance(img, gentle=gentle)

def auto_enhance(img: Image.Image, gentle: bool = False) -> Image.Image:
    """gentle=True → BG flow (skin tone SAFE)."""
    arr = np.array(img.convert("RGB"))

    result = arr.astype(np.float32)
    avg = result.mean(axis=(0, 1))
    deviation = np.abs(avg - avg.mean()) / max(avg.mean(), 1)
    if deviation.max() > 0.08:
        gains = np.clip(avg.mean() / np.maximum(avg, 1), 0.90, 1.10)
        result *= gains
    arr = np.clip(result, 0, 255).astype(np.uint8)

    clip = 1.3 if gentle else 1.8
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
    arr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    if gentle:
        blur = cv2.GaussianBlur(arr, (0, 0), 1.0)
        arr = cv2.addWeighted(arr, 1.12, blur, -0.12, 0)
    else:
        blur = cv2.GaussianBlur(arr, (0, 0), 1.2)
        arr = cv2.addWeighted(arr, 1.25, blur, -0.25, 0)

    return Image.fromarray(arr)

def post_resize_sharpen(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    blur = cv2.GaussianBlur(arr, (0, 0), 0.8)
    sharp = cv2.addWeighted(arr, 1.15, blur, -0.15, 0)
    return Image.fromarray(sharp)

def create_preview(img: Image.Image) -> io.BytesIO:
    if img.mode in ("RGBA", "LA", "PA"):
        img = flatten_on_white(img)
    preview = img.copy()
    preview.thumbnail(PREVIEW_MAX_SIZE, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    preview.convert("RGB").save(buf, format="JPEG", quality=82, optimize=True)
    buf.seek(0)
    return buf

def format_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"

def parse_dimensions(text: str, dpi: int = DPI_DEFAULT) -> Optional[Tuple[int, int]]:
    """Trailing unit regex ko anchor karta hai — bare px kabhi mm/cm nahi banta.
    '3.5x4.5cm' AUR '3.5cm x 4.5cm' dono chalte hain."""
    if not text:
        return None
    text = text.strip().lower()
    m = re.match(r"(\d+(?:\.\d+)?)\s*(?:mm)?\s*x\s*(\d+(?:\.\d+)?)\s*mm", text)
    if m:
        px = lambda v: int(round(float(v) / 25.4 * dpi))
        return px(m.group(1)), px(m.group(2))
    m = re.match(r"(\d+(?:\.\d+)?)\s*(?:cm)?\s*x\s*(\d+(?:\.\d+)?)\s*cm", text)
    if m:
        px = lambda v: int(round(float(v) / 2.54 * dpi))
        return px(m.group(1)), px(m.group(2))
    m = re.match(r"(\d+(?:\.\d+)?)\s*(?:in)?\s*x\s*(\d+(?:\.\d+)?)\s*in(?:ch)?", text)
    if m:
        px = lambda v: int(round(float(v) * dpi))
        return px(m.group(1)), px(m.group(2))
    m = re.match(r"(\d+)\s*(?:px)?\s*x\s*(\d+)\s*(?:px)?", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

def clamp_dims(w: int, h: int) -> Tuple[int, int]:
    return max(MIN_DIM, min(MAX_IMAGE_DIM, w)), max(MIN_DIM, min(MAX_IMAGE_DIM, h))

def parse_size_kb(text: str) -> Optional[int]:
    if not text:
        return None
    text = text.strip().lower()
    m = re.match(r"(\d+(?:\.\d+)?)\s*mb", text)
    if m:
        kb = int(float(m.group(1)) * 1024)
    else:
        digits = re.sub(r"[^\d]", "", text)
        if not digits:
            return None
        kb = int(digits)
    return max(5, min(20480, kb))

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

def smart_resize(img: Image.Image, w: int, h: int, pad_mode: bool) -> Image.Image:
    if pad_mode:
        return ImageOps.pad(img, (w, h),
                            method=Image.Resampling.LANCZOS, color=(255, 255, 255))
    iw, ih = img.size
    target_ratio = w / h
    if iw / ih > target_ratio:
        nw = int(ih * target_ratio)
        x = (iw - nw) // 2
        img = img.crop((x, 0, x + nw, ih))
    else:
        nh = int(iw / target_ratio)
        y = (ih - nh) // 2
        img = img.crop((0, y, iw, y + nh))
    return img.resize((w, h), Image.Resampling.LANCZOS)

def sanitize_md(text: str, maxlen: int = 60) -> str:
    return re.sub(r"[*_`\[\]]", "", text).strip()[:maxlen]

def _content_ratio(img: Image.Image) -> float:
    """Non-white pixel ratio (RGB images only)."""
    g = np.asarray(img.convert("L"))
    return float((g < 235).mean())

# ─────────────────────────────────────────────────────────────────────
# FACE-AWARE PASSPORT CROP v6.5 — crop-in-source-coords
# ─────────────────────────────────────────────────────────────────────
def detect_face_mp(img: Image.Image) -> Optional[Dict[str, Any]]:
    arr = np.array(img.convert("RGB"))
    det_state = run_face_detect(arr)
    if not det_state or not det_state.detections:
        return None
    box = det_state.detections[0].location_data.relative_bounding_box
    ih, iw = arr.shape[:2]
    return {
        "cx": (box.xmin + box.width / 2) * iw,
        "cy": (box.ymin + box.height / 2) * ih,
        "w":  box.width * iw,
        "h":  box.height * ih,
    }

def passport_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    v6.5 REWRITE — crop window SOURCE coordinates mein compute hota hai:
      s  = target_h*0.70 / face_h
      window = (target_w/s × target_h/s) centered so face lands at
               (50%, 45%) of output
    Window ko source bounds se intersect karke sirf overlap piece
    resize hota hai → NO giant intermediates (8192² bomb gone),
    scale>2 cap hata — positioning AB BHI exact, kabhi distort nahi.
    """
    iw, ih = img.size
    try:
        face = detect_face_mp(img)
    except Exception:
        face = None

    if not face or face["h"] <= 0:
        return smart_resize(img, target_w, target_h, pad_mode=False)

    s = (target_h * 0.70) / face["h"]
    cw = target_w / s                    # window size in SOURCE px
    ch = target_h / s
    x0 = face["cx"] - cw / 2.0           # window origin — face → (50%, 45%)
    y0 = face["cy"] - (target_h * 0.45) / s

    # Intersection with source bounds
    ix0, iy0 = max(0.0, x0), max(0.0, y0)
    ix1, iy1 = min(float(iw), x0 + cw), min(float(ih), y0 + ch)

    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    if ix1 > ix0 and iy1 > iy0:
        crop = img.crop((int(ix0), int(iy0), int(ix1), int(iy1)))
        dw = int(round((ix1 - ix0) * s))
        dh = int(round((iy1 - iy0) * s))
        if dw > 0 and dh > 0:
            crop = crop.resize((dw, dh), Image.Resampling.LANCZOS)
            dx = int(round((ix0 - x0) * s))   # white padding auto: shifted paste
            dy = int(round((iy0 - y0) * s))
            canvas.paste(crop, (dx, dy))
    return canvas

# ─────────────────────────────────────────────────────────────────────
# CORE: BACKGROUND CHANGE v6.2 — trimap matting + decontamination
# ─────────────────────────────────────────────────────────────────────
def guided_filter(I: np.ndarray, p: np.ndarray, r: int = 8, eps: float = 1e-3) -> np.ndarray:
    I_f = I.astype(np.float32) / 255.0
    p_f = p.astype(np.float32)
    mean_I  = cv2.boxFilter(I_f, -1, (r, r))
    mean_p  = cv2.boxFilter(p_f, -1, (r, r))
    mean_Ip = cv2.boxFilter(I_f * p_f, -1, (r, r))
    cov_Ip  = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I_f * I_f, -1, (r, r))
    var_I   = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, -1, (r, r))
    mean_b = cv2.boxFilter(b, -1, (r, r))
    return (mean_a * I_f + mean_b).clip(0, 1)

def person_segmentation_replace(img: Image.Image, color_text: str) -> Image.Image:
    """
    MATTING-GRADE: guided refine → trimap (hair survives) →
    decontamination (no halo) → smooth → continuous composite.
    """
    if max(img.size) > BG_MAX_DIM:
        s = BG_MAX_DIM / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)),
                         Image.Resampling.LANCZOS)

    rgb_arr = np.array(img.convert("RGB"))
    h, w = rgb_arr.shape[:2]
    img_f = rgb_arr.astype(np.float32)

    result = run_segmentation(rgb_arr)
    mask_f = result.segmentation_mask.astype(np.float32)
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
    r = max(4, min(h, w) // 100)
    mask_ref = np.clip(guided_filter(gray, mask_f, r=r, eps=1e-4), 0, 1)
    del result, mask_f, gray
    gc.collect()

    core = (mask_ref > 0.75).astype(np.uint8)
    k = max(3, int(min(h, w) / 250)); k += (k + 1) % 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, kernel)
    sure_bg = mask_ref < 0.12

    alpha = np.where(core > 0, 1.0,
            np.where(sure_bg, 0.0, mask_ref)).astype(np.float32)
    del core, sure_bg, mask_ref
    gc.collect()

    alpha = np.power(alpha, 0.85)
    alpha = cv2.medianBlur(alpha, 3)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

    # 🌟 Decontamination — edge pixels se purana bg un-mix
    border = np.concatenate([
        img_f[:12].reshape(-1, 3),  img_f[-12:].reshape(-1, 3),
        img_f[:, :12].reshape(-1, 3), img_f[:, -12:].reshape(-1, 3)])
    bg_orig = np.median(border, axis=0)

    a3 = alpha[..., None]
    semi = (a3 > 0.25) & (a3 < 0.98)
    if np.any(semi):
        a_safe = np.maximum(a3, 0.25)
        fg_est = (img_f - (1.0 - a3) * bg_orig) / a_safe
        img_f[semi[..., 0]] = fg_est[semi[..., 0]]
        del fg_est
    del semi, a3
    gc.collect()

    color = parse_color(color_text)
    a3 = alpha[..., None]
    out = img_f * a3 + np.float32(color) * (1.0 - a3)
    out = np.clip(out, 0, 255).astype(np.uint8)

    del img_f, alpha, a3
    gc.collect()
    return Image.fromarray(out)

# ─────────────────────────────────────────────────────────────────────
# CORE: SIGNATURE v2.1 — auto-downscale + hysteresis + smooth alpha
# ─────────────────────────────────────────────────────────────────────
def extract_signature(img: Image.Image) -> Image.Image:
    # v6.5: bade scans pe bilateral filter bahut slow — 2000px kaafi hai
    if max(img.size) > SIG_MAX_DIM:
        s = SIG_MAX_DIM / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)),
                         Image.Resampling.LANCZOS)

    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)

    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    strong = (gray < 140).astype(np.uint8) * 255
    weak   = (gray < 190).astype(np.uint8) * 255

    num, labels, stats, _ = cv2.connectedComponentsWithStats(weak, 8)
    result_mask = np.zeros_like(strong)
    for lbl in range(1, num):
        comp = (labels == lbl)
        if np.any(comp & (strong > 0)):
            result_mask[comp] = 255

    num, labels, stats, _ = cv2.connectedComponentsWithStats(result_mask, 8)
    min_area = max(8, int(result_mask.size * 0.00003))
    clean = np.zeros_like(result_mask)
    for lbl in range(1, num):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == lbl] = 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=1)

    dist = cv2.distanceTransform((clean > 0).astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.clip(dist * 3.0, 0, 1)
    alpha[clean > 0] = np.maximum(alpha[clean > 0], 0.7)
    alpha = (alpha * 255).astype(np.uint8)

    coords = cv2.findNonZero((alpha > 0).astype(np.uint8))
    if coords is not None:
        x, y, bw, bh = cv2.boundingRect(coords)
        px = max(20, int(bw * 0.12)); py = max(20, int(bh * 0.12))
        alpha = alpha[max(0, y - py):min(alpha.shape[0], y + bh + py),
                      max(0, x - px):min(alpha.shape[1], x + bw + px)]

    out = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8)
    out[..., 0] = 10; out[..., 1] = 10; out[..., 2] = 12
    out[..., 3] = alpha
    return Image.fromarray(out, "RGBA")

def _sig_bg_is_white(img: Image.Image) -> bool:
    """Thread-runner — corner brightness check."""
    arr = np.array(img.convert("RGB"))
    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    return all(v > 190 for v in np.mean(corners, axis=0))

# ─────────────────────────────────────────────────────────────────────
# CORE: COMPRESS (reduce flow — dimension fallback allowed)
# ─────────────────────────────────────────────────────────────────────
def compress_to_kb(img: Image.Image, target_kb: int, fmt: str = "JPEG") -> io.BytesIO:
    buf = io.BytesIO()
    fmt_up = fmt.upper()
    target_b = target_kb * 1024

    if fmt_up == "PDF":
        flatten_on_white(img).save(buf, format="PDF", resolution=DPI_DEFAULT)
        buf.seek(0); return buf

    if fmt_up == "PNG":
        curr, scale = img, 1.0
        while True:
            buf.seek(0); buf.truncate()
            curr.save(buf, format="PNG", optimize=True)
            if buf.tell() <= target_b or scale <= 0.05:
                break
            scale -= 0.08
            curr = img.resize((max(1, int(img.width * scale)),
                               max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
        buf.seek(0); return buf

    src = flatten_on_white(img)
    lo, hi = MIN_QUALITY, MAX_QUALITY
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        buf.seek(0); buf.truncate()
        src.save(buf, format="JPEG", quality=mid, optimize=True)
        if buf.tell() <= target_b:
            best = buf.getvalue(); lo = mid + 1
        else:
            hi = mid - 1

    if best is not None:
        buf.seek(0); buf.truncate(); buf.write(best); buf.seek(0)
        return buf

    lo_s, hi_s = 0.1, 0.9
    while hi_s - lo_s > 0.02:
        mid_s = (lo_s + hi_s) / 2
        simg = img.resize((max(1, int(img.width * mid_s)),
                           max(1, int(img.height * mid_s))), Image.Resampling.LANCZOS)
        buf.seek(0); buf.truncate()
        flatten_on_white(simg).save(buf, format="JPEG", quality=80, optimize=True)
        if buf.tell() <= target_b:
            best = buf.getvalue(); lo_s = mid_s
        else:
            hi_s = mid_s
    if best:
        buf.seek(0); buf.truncate(); buf.write(best)
    buf.seek(0)
    return buf

# ─────────────────────────────────────────────────────────────────────
# 🎯 CORE: EXACT SIZE MATCH — same resolution GUARANTEED
# ─────────────────────────────────────────────────────────────────────
def _jpeg_encode(src: Image.Image, q: int) -> bytes:
    b = io.BytesIO()
    kw = dict(format="JPEG", quality=q, optimize=True)
    if q >= 90:
        kw["subsampling"] = 0
    src.save(b, **kw)
    return b.getvalue()

def match_file_size_kb(img: Image.Image, target_kb: int,
                       fmt: str = "JPEG") -> Tuple[io.BytesIO, Dict[str, Any]]:
    target = target_kb * 1024
    src = flatten_on_white(img)

    if fmt.upper() == "PNG":
        # v6.4 FIX retained: level 0→9 — pehla fit = target ke SABSE KAREEB
        best_data, warn, reduced = None, None, False
        for lvl in range(0, 10):
            b = io.BytesIO()
            src.save(b, format="PNG", compress_level=lvl)
            if b.tell() <= target:
                best_data = b.getvalue(); break
        if best_data is None:
            for colors in (256, 192, 128, 96, 64, 48, 32, 16):
                q = src.quantize(colors=colors, dither=Image.FLOYDSTEINBERG)
                b = io.BytesIO()
                q.save(b, format="PNG", optimize=True)
                best_data = b.getvalue()
                reduced = True
                if b.tell() <= target:
                    break
            if len(best_data) > target:
                warn = "min"
        return io.BytesIO(best_data), {"warn": warn,
                                       "quality": "colors-reduced" if reduced else "lossless"}

    # JPEG Phase A: quality binary search
    best_data = None
    lo, hi = MIN_QUALITY, MAX_QUALITY
    while lo <= hi:
        mid = (lo + hi) // 2
        data = _jpeg_encode(src, mid)
        sz = len(data)
        if abs(sz - target) <= target * 0.02:
            warn = None if mid >= 30 else "quality"
            return io.BytesIO(data), {"quality": mid, "warn": warn}
        if sz < target:
            best_data = data
            lo = mid + 1
        else:
            hi = mid - 1

    if best_data is None:
        data = _jpeg_encode(src, MIN_QUALITY)
        return io.BytesIO(data), {"quality": MIN_QUALITY, "warn": "min"}

    # JPEG Phase B: size badhana → grain noise (≤4MP)
    best_up = None
    if src.width * src.height <= NOISE_MAX_PIXELS:
        rng = np.random.default_rng(42)
        noise1 = rng.standard_normal((src.height, src.width, 3)).astype(np.float32)
        arr0 = np.array(src).astype(np.float32)
        lo_s, hi_s = 0.3, 10.0
        while hi_s - lo_s > 0.05:
            mid_s = (lo_s + hi_s) / 2
            arr = np.clip(arr0 + noise1 * mid_s, 0, 255).astype(np.uint8)
            data = _jpeg_encode(Image.fromarray(arr), MAX_QUALITY)
            if len(data) <= target:
                best_up = data; lo_s = mid_s
            else:
                hi_s = mid_s
        if best_up is not None and abs(len(best_up) - target) <= target * 0.05:
            return io.BytesIO(best_up), {"quality": MAX_QUALITY,
                                         "noise": round(lo_s, 1), "warn": None}

    if best_up is not None:
        return io.BytesIO(best_up), {"quality": MAX_QUALITY, "warn": "max"}
    return io.BytesIO(_jpeg_encode(src, MAX_QUALITY)), {"quality": MAX_QUALITY,
                                                        "warn": "max"}

# ─────────────────────────────────────────────────────────────────────
# 🖨 CORE: PRINT SHEET + empty-sheet guard
# ─────────────────────────────────────────────────────────────────────
def make_photo_sheet(img: Image.Image, dpi: int = 300,
                     cols: int = 4, rows: int = 2,
                     gap_mm: float = 2.0) -> Image.Image:
    cw, ch = int(6.0 * dpi), int(4.0 * dpi)
    gap = int(gap_mm / 25.4 * dpi)
    cell_w = (cw - (cols + 1) * gap) // cols
    cell_h = (ch - (rows + 1) * gap) // rows

    photo = passport_crop(img, cell_w, cell_h)

    # 🛡️ 90%+ white = crop fail → center-crop fallback
    if _content_ratio(photo) < 0.10:
        logger.warning("passport_crop near-empty canvas — center-crop fallback")
        photo = smart_resize(img, cell_w, cell_h, pad_mode=False)

    sheet = Image.new("RGB", (cw, ch), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for r_i in range(rows):
        for c_i in range(cols):
            x = gap + c_i * (cell_w + gap)
            y = gap + r_i * (cell_h + gap)
            sheet.paste(photo, (x, y))
            d.rectangle([x, y, x + cell_w, y + cell_h],
                        outline=(185, 185, 185), width=1)
    return sheet

def _sheet_encode(sheet: Image.Image) -> Tuple[bytes, bytes]:
    jb = io.BytesIO()
    sheet.save(jb, format="JPEG", quality=MAX_QUALITY, dpi=(300, 300), optimize=True)
    pb = io.BytesIO()
    sheet.save(pb, format="PDF", resolution=300)
    return jb.getvalue(), pb.getvalue()

# ─────────────────────────────────────────────────────────────────────
# SAVE / VALIDATE
# ─────────────────────────────────────────────────────────────────────
def save_image(img: Image.Image, fmt: str, dpi_val: int) -> io.BytesIO:
    buf = io.BytesIO()
    f = fmt.upper()
    if f == "JPEG":
        flatten_on_white(img).save(buf, format="JPEG", quality=MAX_QUALITY,
                                   dpi=(dpi_val, dpi_val), optimize=True)
    elif f == "PNG":
        img.save(buf, format="PNG", dpi=(dpi_val, dpi_val))
    elif f == "PDF":
        flatten_on_white(img).save(buf, format="PDF", resolution=dpi_val)
    buf.seek(0)
    return buf

VALID_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF", "MPO", "HEIF", "AVIF"}

def validate_image_bytes(data: bytes) -> bool:
    try:
        return Image.open(io.BytesIO(data)).format in VALID_FORMATS
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────────────
# TELEGRAM HELPERS
# ─────────────────────────────────────────────────────────────────────
async def safe_reply(update: Update, text: str, reply_markup=None, parse_mode="Markdown"):
    target = update.message or (update.callback_query.message if update.callback_query else None)
    if target:
        await target.reply_text(text, reply_markup=reply_markup, parse_mode=parse_mode)

# ─────────────────────────────────────────────────────────────────────
# SERVICE-MESSAGE MANAGER — ek hi interactive message, ghost impossible
# ─────────────────────────────────────────────────────────────────────
async def _svc_delete(ctx, chat_id: int):
    mid = ctx.user_data.pop("svc_msg_id", None)
    if mid:
        try:
            await ctx.bot.delete_message(chat_id, mid)
        except Exception:
            pass

async def svc_prompt(update: Update, ctx, text: str,
                     reply_markup=None, parse_mode="Markdown"):
    chat_id = update.effective_chat.id
    await _svc_delete(ctx, chat_id)
    msg = await ctx.bot.send_message(chat_id, text,
                                     reply_markup=reply_markup,
                                     parse_mode=parse_mode)
    ctx.user_data["svc_msg_id"] = msg.message_id
    return msg

async def svc_prompt_photo(update: Update, ctx, photo, caption: str,
                           reply_markup=None):
    chat_id = update.effective_chat.id
    await _svc_delete(ctx, chat_id)
    msg = await ctx.bot.send_photo(chat_id, photo=photo, caption=caption,
                                   reply_markup=reply_markup,
                                   parse_mode="Markdown")
    ctx.user_data["svc_msg_id"] = msg.message_id
    return msg

async def send_main_menu(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await svc_prompt(update, ctx, t("main_menu", ctx), main_menu_kb(ctx))

async def get_image(update: Update):
    """PIL Image | 'too_large' | 'invalid' | None. Decode thread mein (v6.5)."""
    msg = update.message
    fobj = (msg.photo[-1] if msg.photo else None) or msg.document
    if not fobj:
        return None
    if (getattr(fobj, "file_size", 0) or 0) > MAX_FILE_SIZE_MB * 1024 * 1024:
        return "too_large"
    tg_file = await fobj.get_file()
    raw = io.BytesIO()
    await tg_file.download_to_memory(raw)
    raw.seek(0)
    data = raw.read()
    if not validate_image_bytes(data):
        return "invalid"
    try:
        img = await asyncio.to_thread(_decode_image, data)
    except Exception:
        return "invalid"
    return img

def log_state(uid: int, state: str, action: str):
    logger.info(f"STATE={state} | {action}")

def _store_history(ctx, data: bytes, filename: str):
    old = ctx.user_data.pop("hist_buf", None)
    if isinstance(old, SecureBuffer):
        old.wipe()
    ctx.user_data["hist_buf"]  = SecureBuffer(data)
    ctx.user_data["hist_name"] = filename

async def deliver_result(update: Update, ctx, chat_msg, data: bytes, filename: str,
                         dims: Tuple[int, int], done_key: str):
    size_str = format_size(len(data))
    caption = (f"{t(done_key, ctx)}\n"
               f"📏 `{dims[0]}×{dims[1]}px` | 📦 `{size_str}`\n\n"
               f"{t('reminder', ctx)}")
    _store_history(ctx, data, filename)
    await asyncio.to_thread(bump_op_count, update.effective_user.id)
    await chat_msg.reply_document(document=io.BytesIO(data), filename=filename,
                                  caption=caption, parse_mode="Markdown")

# ─────────────────────────────────────────────────────────────────────
# KEYBOARDS
# ─────────────────────────────────────────────────────────────────────
def main_menu_kb(ctx) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(t("bg_change",   ctx), callback_data="bg_change")],
        [InlineKeyboardButton(t("resize",      ctx), callback_data="resize")],
        [InlineKeyboardButton(t("signature",   ctx), callback_data="signature")],
        [InlineKeyboardButton(t("size_match",  ctx), callback_data="size_match")],
        [InlineKeyboardButton(t("print_sheet", ctx), callback_data="print_sheet")],
    ])

def bg_color_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚪ White",       callback_data="col_white"),
         InlineKeyboardButton("🟡 Off-White",   callback_data="col_#f5f0e8")],
        [InlineKeyboardButton("🔵 Light Blue",  callback_data="col_#add8e6"),
         InlineKeyboardButton("🟢 Light Green", callback_data="col_#90ee90")],
        [InlineKeyboardButton("🔴 Red",         callback_data="col_red"),
         InlineKeyboardButton("⬜ Light Grey",  callback_data="col_#d3d3d3")],
        [InlineKeyboardButton("🎨 Custom Color", callback_data="col_custom")],
    ])

def confirm_kb(ok_cb: str, retry_cb: str, ctx) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(t("looks_ok", ctx), callback_data=ok_cb),
        InlineKeyboardButton(t("retry",    ctx), callback_data=retry_cb)]])

def format_kb(include_pdf: bool = True) -> InlineKeyboardMarkup:
    row = [InlineKeyboardButton("JPEG", callback_data="fmt_JPEG"),
           InlineKeyboardButton("PNG",  callback_data="fmt_PNG")]
    if include_pdf:
        row.append(InlineKeyboardButton("PDF", callback_data="fmt_PDF"))
    return InlineKeyboardMarkup([row])

def size_fmt_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("JPEG ⭐", callback_data="fmt_JPEG"),
        InlineKeyboardButton("PNG",    callback_data="fmt_PNG")]])

def resize_mode_kb(ctx) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📋 Preset Sizes",     callback_data="resize_preset")],
        [InlineKeyboardButton("📐 Custom Dimensions", callback_data="resize_custom")],
        [InlineKeyboardButton("📦 Reduce File Size",  callback_data="resize_reduce")]])

async def preset_kb() -> InlineKeyboardMarkup:
    presets = (await asyncio.to_thread(db_get_presets))[:40]
    if not presets:
        return InlineKeyboardMarkup([[
            InlineKeyboardButton("📐 Custom Dimensions", callback_data="preset_custom")]])
    rows, i = [], 0
    while i < len(presets):
        rows.append([InlineKeyboardButton(p["label"], callback_data=f"preset_{p['id']}")
                     for p in presets[i:i + 2]])
        i += 2
    rows.append([InlineKeyboardButton("📐 Custom Dimensions", callback_data="preset_custom")])
    return InlineKeyboardMarkup(rows)

def size_option_kb(ctx) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(t("size_by_kb",   ctx), callback_data="sizeopt_kb")],
        [InlineKeyboardButton(t("size_by_dims", ctx), callback_data="sizeopt_save")]])

def sig_format_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("JPEG (white bg)",   callback_data="fmt_JPEG"),
        InlineKeyboardButton("PNG (transparent)", callback_data="fmt_PNG"),
        InlineKeyboardButton("PDF",               callback_data="fmt_PDF")]])

# ─────────────────────────────────────────────────────────────────────
# COMMAND HANDLERS
# ─────────────────────────────────────────────────────────────────────
async def _ensure_user(update: Update, ctx):
    """v6.5 FIX: pehli baar aaye user ka DB row TURANT banta hai
    (pehle sirf 2nd /start pe banta tha — stats corrupt ho rahe the)."""
    uid = update.effective_user.id
    _known_real_ids.add(uid)
    if "hinglish" not in ctx.user_data:
        ctx.user_data.update(await asyncio.to_thread(get_user_prefs, uid))
        await asyncio.to_thread(upsert_user, uid)

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    await _ensure_user(update, ctx)
    cleanup_session(ctx)
    await send_main_menu(update, ctx)
    return S.SELECT_ACTION

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await safe_reply(update, t("help_text", ctx))

async def cmd_privacy(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await safe_reply(update, t("privacy_notice", ctx))

async def cmd_hinglish(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    curr = ctx.user_data.get("hinglish", False)
    ctx.user_data["hinglish"] = not curr
    await _ensure_user(update, ctx)
    await asyncio.to_thread(save_user_pref, update.effective_user.id, "hinglish", int(not curr))
    await safe_reply(update, t("lang_hi" if not curr else "lang_en", ctx))

async def cmd_strict(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    curr = ctx.user_data.get("strict", True)
    ctx.user_data["strict"] = not curr
    await _ensure_user(update, ctx)
    await asyncio.to_thread(save_user_pref, update.effective_user.id, "strict", int(not curr))
    await safe_reply(update, t("strict_on" if not curr else "strict_off", ctx))

async def cmd_dpi(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if ctx.args and ctx.args[0].isdigit() and int(ctx.args[0]) in ALLOWED_DPIS:
        v = int(ctx.args[0])
        ctx.user_data["dpi"] = v
        await _ensure_user(update, ctx)
        await asyncio.to_thread(save_user_pref, update.effective_user.id, "dpi", v)
        await safe_reply(update, t("dpi_set", ctx) + f"`{v}`")
    else:
        await safe_reply(update, t("dpi_usage", ctx))

async def cmd_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    data = secure_peek(ctx, "hist_buf")
    if not data:
        await safe_reply(update, t("no_history", ctx))
        return
    name = ctx.user_data.get("hist_name", "output.jpg")
    target = update.message or update.callback_query.message
    await target.reply_document(document=io.BytesIO(data), filename=name,
                                caption=t("history", ctx))

async def cmd_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    cleanup_session(ctx)
    await safe_reply(update, t("cancel", ctx))
    await send_main_menu(update, ctx)
    return S.SELECT_ACTION

async def cmd_mystats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    hid = hash_uid(update.effective_user.id)
    def _q():
        with get_db() as conn:
            return conn.execute(
                "SELECT total_ops,dpi FROM users WHERE hashed_uid=?", (hid,)).fetchone()
    row = await asyncio.to_thread(_q)
    if not row:
        await safe_reply(update, "📊 No stats yet. Use the bot first!")
        return
    await safe_reply(update,
        f"📊 *Your Stats*\n\nTotal operations: `{row['total_ops']}`\n"
        f"Output DPI: `{row['dpi']}`\n\n"
        f"_Anonymous hash only. No timestamps, names, or Telegram ID._")

# ─────────────────────────────────────────────────────────────────────
# ADMIN
# ─────────────────────────────────────────────────────────────────────
async def cmd_admin(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        await safe_reply(update, "❌ Access denied."); return
    stats = await asyncio.to_thread(get_bot_stats)
    up = int(time.time() - BOT_START_TIME)
    heic = "✅" if HEIC_OK else "❌ pip install pillow-heif"
    mp_s = "✅" if MP_OK else "❌ pip install mediapipe"
    ws = "waitress ✓" if WAITRESS_OK else "flask-dev"
    await safe_reply(update,
        f"🛠 *Admin Panel — {VERSION}*\n\n"
        f"Uptime: `{up//3600}h {(up%3600)//60}m` | 🌐 {ws}\n"
        f"👥 Users: `{stats['total_users']}` | 📊 Ops: `{stats['total_ops']}`\n"
        f"🧠 MediaPipe: {mp_s} | 📱 HEIC: {heic}\n\n"
        f"_No personal data stored. Real IDs hashed._")

async def cmd_broadcast(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        await safe_reply(update, "❌ Access denied."); return
    if not ctx.args:
        await safe_reply(update, "Usage: `/broadcast Your message here`"); return
    message = " ".join(ctx.args)
    real_ids = list(_known_real_ids)
    if not real_ids:
        await safe_reply(update, "⚠️ No active users in current session."); return
    status = await update.message.reply_text(f"📡 Broadcasting to {len(real_ids)} users...")
    sent = failed = 0
    for rid in real_ids:
        for _attempt in range(2):
            try:
                await ctx.bot.send_message(rid, f"📢 Announcement\n\n{message}")
                sent += 1
                break
            except RetryAfter as e:
                await asyncio.sleep(e.retry_after + 1)
            except Exception:
                failed += 1
                break
        await asyncio.sleep(0.05)
    def _log():
        with get_db() as conn:
            conn.execute("INSERT INTO broadcasts(message,sent_at) VALUES(?,?)",
                         (message, datetime.now(timezone.utc).isoformat()))
    await asyncio.to_thread(_log)
    await status.edit_text(f"✅ Sent: `{sent}` | Failed: `{failed}`")

async def cmd_listpresets(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        await safe_reply(update, "❌ Access denied."); return
    presets = await asyncio.to_thread(db_list_all_presets)
    if not presets:
        await safe_reply(update, "📋 No presets yet."); return
    lines = ["📋 *All Presets*\n"]
    for p in presets:
        st = "✅" if p["active"] else "❌"
        lines.append(f"{st} `ID:{p['id']}` — {sanitize_md(p['label'])} — `{p['width_px']}×{p['height_px']}px`")
    lines.append("\n`/addpreset Label | w | h`\n`/editpreset ID | Label | w | h`\n`/delpreset ID`")
    await safe_reply(update, "\n".join(lines))

async def cmd_addpreset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        await safe_reply(update, "❌ Access denied."); return
    raw = " ".join(ctx.args) if ctx.args else ""
    parts = [p.strip() for p in raw.split("|")]
    if len(parts) != 3:
        await safe_reply(update, "📝 *Format:* `/addpreset Label | width | height`\n"
                                 "*Example:* `/addpreset Railway 2025 | 413 | 531`"); return
    label = sanitize_md(parts[0], 40)
    try:
        w, h = int(parts[1]), int(parts[2])
        if not (MIN_DIM <= w <= 5000 and MIN_DIM <= h <= 5000):
            raise ValueError
    except ValueError:
        await safe_reply(update, "❌ Width/height must be numbers (16–5000 px)."); return
    new_id = await asyncio.to_thread(db_add_preset, label, w, h)
    await safe_reply(update, f"✅ *Preset added!*\n`ID:{new_id}` — {label} — `{w}×{h}px`")

async def cmd_editpreset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        await safe_reply(update, "❌ Access denied."); return
    raw = " ".join(ctx.args) if ctx.args else ""
    parts = [p.strip() for p in raw.split("|")]
    if len(parts) != 4:
        await safe_reply(update, "📝 *Format:* `/editpreset ID | Label | width | height`"); return
    try:
        pid, w, h = int(parts[0]), int(parts[2]), int(parts[3])
        if not (MIN_DIM <= w <= 5000 and MIN_DIM <= h <= 5000):
            raise ValueError
    except ValueError:
        await safe_reply(update, "❌ Invalid values."); return
    preset = await asyncio.to_thread(db_get_preset_by_id, pid)
    if not preset:
        await safe_reply(update, f"❌ ID `{pid}` not found."); return
    label = sanitize_md(parts[1], 40)
    await asyncio.to_thread(db_edit_preset, pid, label, w, h)
    await safe_reply(update, f"✅ Updated!\n{sanitize_md(preset['label'])} → {label} `{w}×{h}px`")

async def cmd_delpreset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id not in ADMIN_IDS:
        await safe_reply(update, "❌ Access denied."); return
    if not ctx.args or not ctx.args[0].isdigit():
        await safe_reply(update, "📝 *Format:* `/delpreset ID`"); return
    pid = int(ctx.args[0])
    preset = await asyncio.to_thread(db_get_preset_by_id, pid)
    if not preset:
        await safe_reply(update, f"❌ ID `{pid}` not found."); return
    await asyncio.to_thread(db_delete_preset, pid)
    await safe_reply(update, f"🗑 Deleted: {sanitize_md(preset['label'])}")

# ─────────────────────────────────────────────────────────────────────
# MAIN FLOW: SELECT — lock-fail/rate-limit pe return None = STATE PRESERVED
# ─────────────────────────────────────────────────────────────────────
async def select_action(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    q = update.callback_query
    uid = update.effective_user.id
    log_state(uid, "SELECT_ACTION", q.data)
    await _ensure_user(update, ctx)

    # Lock held → koi bhi menu action alert-only, chalu operation untouched
    if ctx.user_data.get("_lock"):
        await q.answer(t("processing_lock", ctx), show_alert=True)
        return None

    if not check_rate_limit(uid):
        await q.answer(t("rate_limit", ctx), show_alert=True)
        return None

    if q.data == "resize":
        await q.answer()
        await svc_prompt(update, ctx, t("resize", ctx), resize_mode_kb(ctx))
        return S.RESIZE_MODE

    op_map = {
        "bg_change":   ("send_photo",       S.BG_WAIT_PHOTO),
        "signature":   ("send_photo",       S.SIG_WAIT_PHOTO),
        "size_match":  ("size_send_photo",  S.SIZE_WAIT_PHOTO),
        "print_sheet": ("sheet_send_photo", S.SHEET_WAIT_PHOTO),
    }
    if q.data in op_map:
        if not acquire_lock(ctx):
            await q.answer(t("processing_lock", ctx), show_alert=True)
            return None
        await q.answer()
        _start_op(update, ctx)
        prompt_key, next_state = op_map[q.data]
        await svc_prompt(update, ctx, t(prompt_key, ctx))
        return next_state

    await q.answer()
    return None

# ─────────────────────────────────────────────────────────────────────
# BACKGROUND CHANGE FLOW
# ─────────────────────────────────────────────────────────────────────
async def bg_wait_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    uid = update.effective_user.id
    _known_real_ids.add(uid)
    if not check_rate_limit(uid):
        await safe_reply(update, t("rate_limit", ctx)); return S.BG_WAIT_PHOTO
    img = await get_image(update)
    if img == "too_large":
        await safe_reply(update, t("file_too_large", ctx)); return S.BG_WAIT_PHOTO
    if img in ("invalid", None):
        await safe_reply(update, t("invalid_file" if img == "invalid" else "no_photo", ctx))
        return S.BG_WAIT_PHOTO
    fb, fw, img = await asyncio.to_thread(_preflight, img, ctx, True)
    await safe_reply(update, fb)
    if fw:
        await safe_reply(update, t(fw, ctx))
    await a_secure_store(ctx, "bg_img", img)
    img = None; gc.collect()
    await svc_prompt(update, ctx, t("color_choose", ctx), bg_color_kb())
    return S.BG_WAIT_COLOR

async def bg_wait_color(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    color_text = None
    if update.callback_query:
        await update.callback_query.answer()
        data = update.callback_query.data
        if data == "col_custom":
            await svc_prompt(update, ctx, t("custom_color_prompt", ctx))
            return S.BG_WAIT_COLOR
        if data.startswith("col_"):
            color_text = data[4:]
    elif update.message:
        raw = (update.message.text or "").strip()
        if not raw:
            return S.BG_WAIT_COLOR
        if not validate_color(raw):
            # error + keyboard dono — buttons kabhi khoye nahi
            await svc_prompt(update, ctx,
                f"❌ `{sanitize_md(raw, 30)}` invalid. Try: `white`, `#FF0000`",
                bg_color_kb())
            return S.BG_WAIT_COLOR
        color_text = raw
    if not color_text:
        return S.BG_WAIT_COLOR

    img = await a_secure_load(ctx, "bg_img")
    if not img:
        await svc_prompt(update, ctx, t("error", ctx))
        cleanup_session(ctx); await send_main_menu(update, ctx)
        return S.SELECT_ACTION

    proc = await svc_prompt(update, ctx, t("processing", ctx))
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(person_segmentation_replace, img, color_text),
            timeout=PROCESSING_TIMEOUT)
        img = None; gc.collect()
        await a_secure_store(ctx, "bg_result", result)
        result = None; gc.collect()
    except RuntimeError:
        img = None; gc.collect()
        await proc.edit_text(t("ai_unavailable", ctx))
        cleanup_session(ctx); await send_main_menu(update, ctx)
        return S.SELECT_ACTION
    except asyncio.TimeoutError:
        img = None; gc.collect()
        await proc.edit_text(t("timeout_err", ctx))
        cleanup_session(ctx); await send_main_menu(update, ctx)
        return S.SELECT_ACTION
    except Exception as e:
        img = None; gc.collect()
        logger.error(f"bg_wait_color: {e}", exc_info=True)
        await proc.edit_text(t("error", ctx))
        cleanup_session(ctx); await send_main_menu(update, ctx)
        return S.SELECT_ACTION

    res = await a_secure_load(ctx, "bg_result")
    if not res:
        await svc_prompt(update, ctx, t("error", ctx))
        cleanup_session(ctx); await send_main_menu(update, ctx)
        return S.SELECT_ACTION
    preview = await asyncio.to_thread(create_preview, res)
    await a_secure_store(ctx, "bg_result", res)
    res = None; gc.collect()

    await svc_prompt_photo(update, ctx, preview,
                           f"{t('preview', ctx)}\n\n{t('bg_warning', ctx)}",
                           confirm_kb("bg_ok", "bg_retry", ctx))
    return S.BG_PREVIEW

async def bg_preview(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query; await q.answer()
    if q.data == "bg_ok":
        try:
            await q.edit_message_caption(caption=t("format_choose", ctx),
                                         reply_markup=format_kb(), parse_mode="Markdown")
        except BadRequest:
            await svc_prompt(update, ctx, t("format_choose", ctx), format_kb())
        return S.BG_WAIT_FORMAT
    elif q.data == "bg_retry":
        secure_wipe_all(ctx, ["bg_result"])
        await svc_prompt(update, ctx, t("send_photo", ctx))
        return S.BG_WAIT_PHOTO
    return S.BG_PREVIEW

async def bg_wait_format(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query; await q.answer()
    fmt = q.data.replace("fmt_", "")
    result = await a_secure_load(ctx, "bg_result")
    if not result:
        await svc_prompt(update, ctx, t("error", ctx))
        cleanup_session(ctx); await send_main_menu(update, ctx)
        return S.SELECT_ACTION
    dims = result.size
    try:
        buf = await asyncio.to_thread(save_image, result, fmt,
                                      ctx.user_data.get("dpi", DPI_DEFAULT))
        data = buf.getvalue(); buf.close()
        result = None; gc.collect()
        await deliver_result(update, ctx, q.message, data,
                             f"output.{fmt.lower()}", dims, "bg_done")
        data = None; gc.collect()
    except Exception as e:
        result = None; gc.collect()
        logger.error(f"bg_wait_format: {e}", exc_info=True)
        await safe_reply(update, t("error", ctx))
    finally:
        cleanup_session(ctx)
        await send_main_menu(update, ctx)
    return S.SELECT_ACTION

# ─────────────────────────────────────────────────────────────────────
# RESIZE FLOW
# ─────────────────────────────────────────────────────────────────────
async def resize_mode(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    q = update.callback_query
    # 🌟 v6.5 FIX: lock check PEHLE, answer BAAD MEIN — double-answer
    # BadRequest kabhi nahi hoga, alert user ko actually dikhega
    if q.data in ("resize_preset", "resize_custom"):
        if not acquire_lock(ctx):
            await q.answer(t("processing_lock", ctx), show_alert=True)
            return None
        await q.answer()
        ctx.user_data["_resize_mode"] = "preset" if q.data == "resize_preset" else "custom"
        _start_op(update, ctx)
        await svc_prompt(update, ctx, t("send_photo", ctx))
        return S.CUSTOM_WAIT_PHOTO
    if q.data == "resize_reduce":
        if not acquire_lock(ctx):
            await q.answer(t("processing_lock", ctx), show_alert=True)
            return None
        await q.answer()
        _start_op(update, ctx)
        await svc_prompt(update, ctx, t("reduce_send_photo", ctx))
        return S.REDUCE_WAIT_PHOTO
    await q.answer()
    return S.RESIZE_MODE

async def custom_wait_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    uid = update.effective_user.id
    _known_real_ids.add(uid)
    if not check_rate_limit(uid):
        await safe_reply(update, t("rate_limit", ctx)); return S.CUSTOM_WAIT_PHOTO
    img = await get_image(update)
    if img == "too_large":
        await safe_reply(update, t("file_too_large", ctx)); return S.CUSTOM_WAIT_PHOTO
    if img in ("invalid", None):
        await safe_reply(update, t("invalid_file" if img == "invalid" else "no_photo", ctx))
        return S.CUSTOM_WAIT_PHOTO
    fb, _fw, img = await asyncio.to_thread(_preflight, img, ctx, False)
    await safe_reply(update, fb)
    await a_secure_store(ctx, "resize_img", img)
    img = None; gc.collect()
    if ctx.user_data.get("_resize_mode") == "preset":
        await svc_prompt(update, ctx, t("select_preset", ctx), await preset_kb())
        return S.CUSTOM_SELECT_PRESET
    await svc_prompt(update, ctx, t("dimensions", ctx))
    return S.CUSTOM_WAIT_DIMS

def _do_resize(ctx, img: Image.Image, w: int, h: int) -> Image.Image:
    """Thread-runner. Preset → passport_crop + empty-guard; custom → smart_resize."""
    w, h = clamp_dims(w, h)
    orig_pixels = img.size[0] * img.size[1]
    if ctx.user_data.get("_resize_mode") == "preset":
        try:
            result = passport_crop(img, w, h)
            if _content_ratio(result) < 0.10:
                result = smart_resize(img, w, h, pad_mode=True)
        except Exception:
            result = smart_resize(img, w, h,
                                  pad_mode=bool(ctx.user_data.get("strict", True)))
    else:
        result = smart_resize(img, w, h,
                              pad_mode=bool(ctx.user_data.get("strict", True)))
    if (w * h) < orig_pixels:
        result = post_resize_sharpen(result)
    return result

async def custom_select_preset(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query; await q.answer()
    if q.data == "preset_custom":
        await svc_prompt(update, ctx, t("dimensions", ctx))
        return S.CUSTOM_WAIT_DIMS
    try:
        pid = int(q.data.replace("preset_", ""))
    except ValueError:
        return S.CUSTOM_SELECT_PRESET
    preset = await asyncio.to_thread(db_get_preset_by_id, pid)
    if not preset:
        await svc_prompt(update, ctx, "❌ Preset not found.")
        cleanup_session(ctx); await send_main_menu(update, ctx)
        return S.SELECT_ACTION
    w, h, label = preset["width_px"], preset["height_px"], sanitize_md(preset["label"])
    img = await a_secure_load(ctx, "resize_img")
    if not img:
        await svc_prompt(update, ctx, t("error", ctx))
        cleanup_session(ctx); await send_main_menu(update, ctx)
        return S.SELECT_ACTION
    result = await asyncio.to_thread(_do_resize, ctx, img, w, h)
    img = None; gc.collect()
    preview = await asyncio.to_thread(create_preview, result)
    await a_secure_store(ctx, "resize_result", result)
    result = None; gc.collect()
    await svc_prompt_photo(update, ctx, preview,
        f"{t('preview', ctx)}\n📋 *{label}*\n📏 `{w}×{h}px`\n🎯 Face auto-positioned (govt spec)",
        confirm_kb("resize_ok", "resize_retry", ctx))
    return S.CUSTOM_PREVIEW

async def custom_wait_dims(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text or ""
    dpi = ctx.user_data.get("dpi", DPI_DEFAULT)
    dims = parse_dimensions(text, dpi)
    if not dims:
        await svc_prompt(update, ctx, t("dimensions", ctx)); return S.CUSTOM_WAIT_DIMS
    w, h = clamp_dims(*dims)
    img = await a_secure_load(ctx, "resize_img")
    if not img:
        await svc_prompt(update, ctx, t("error", ctx))
        cleanup_session(ctx); await send_main_menu(update, ctx)
        return S.SELECT_ACTION
    result = await asyncio.to_thread(_do_resize, ctx, img, w, h)
    img = None; gc.collect()
    preview = await asyncio.to_thread(create_preview, result)
    await a_secure_store(ctx, "resize_result", result)
    result = None; gc.collect()
    await svc_prompt_photo(update, ctx, preview,
        f"{t('preview', ctx)}\n📏 `{w}×{h}px`",
        confirm_kb("resize_ok", "resize_retry", ctx))
    return S.CUSTOM_PREVIEW

async def custom_preview(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query; await q.answer()
    if q.data == "resize_ok":
        try:
            await q.edit_message_caption(caption=t("size_option", ctx),
                                         reply_markup=size_option_kb(ctx),
                                         parse_mode="Markdown")
        except BadRequest:
            await svc_prompt(update, ctx, t("size_option", ctx), size_option_kb(ctx))
        return S.CUSTOM_SIZE_OPT
    elif q.data == "resize_retry":
        secure_wipe_all(ctx, ["resize_result"])
        await svc_prompt(update, ctx, t("send_photo", ctx))
        return S.CUSTOM_WAIT_PHOTO
    return S.CUSTOM_PREVIEW

async def custom_size_opt(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query; await q.answer()
    if q.data == "sizeopt_kb":
        await svc_prompt(update, ctx, t("enter_kb", ctx))
        return S.CUSTOM_WAIT_KB
    elif q.data == "sizeopt_save":
        try:
            await q.edit_message_caption(caption=t("format_choose", ctx),
                                         reply_markup=format_kb(), parse_mode="Markdown")
        except BadRequest:
            await svc_prompt(update, ctx, t("format_choose", ctx), format_kb())
        return S.CUSTOM_WAIT_FORMAT
    return S.CUSTOM_SIZE_OPT

async def custom_wait_kb(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    val = parse_size_kb(update.message.text or "")
    if not val:
        await svc_prompt(update, ctx, t("enter_kb", ctx)); return S.CUSTOM_WAIT_KB
    ctx.user_data["target_kb"] = val
    await svc_prompt(update, ctx, t("format_choose", ctx), format_kb())
    return S.CUSTOM_WAIT_FORMAT

async def custom_wait_format(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query; await q.answer()
    fmt = q.data.replace("fmt_", "")
    result = await a_secure_load(ctx, "resize_result")
    if not result:
        await svc_prompt(update, ctx, t("error", ctx))
        cleanup_session(ctx); await send_main_menu(update, ctx)
        return S.SELECT_ACTION
    dims = result.size
    try:
        target_kb = ctx.user_data.get("target_kb")
        if target_kb:
            buf = await asyncio.wait_for(
                asyncio.to_thread(compress_to_kb, result, target_kb, fmt),
                timeout=PROCESSING_TIMEOUT)
        else:
            buf = await asyncio.to_thread(save_image, result, fmt,
                                          ctx.user_data.get("dpi", DPI_DEFAULT))
        data = buf.getvalue(); buf.close()
        result = None; gc.collect()
        await deliver_result(update, ctx, q.message, data,
                             f"output.{fmt.lower()}", dims, "resize_done")
        data = None; gc.collect()
    except Exception as e:
        result = None; gc.collect()
        logger.error(f"custom_wait_format: {e}", exc_info=True)
        await safe_reply(update, t("error", ctx))
    finally:
        cleanup_session(ctx)
        await send_main_menu(update, ctx)
    return S.SELECT_ACTION

# ─────────────────────────────────────────────────────────────────────
# REDUCE SIZE FLOW
# ─────────────────────────────────────────────────────────────────────
async def reduce_wait_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    uid = update.effective_user.id
    _known_real_ids.add(uid)
    if not check_rate_limit(uid):
        await safe_reply(update, t("rate_limit", ctx)); return S.REDUCE_WAIT_PHOTO
    img = await get_image(update)
    if img == "too_large":
        await safe_reply(update, t("file_too_large", ctx)); return S.REDUCE_WAIT_PHOTO
    if img in ("invalid", None):
        await safe_reply(update, t("invalid_file" if img == "invalid" else "no_photo", ctx))
        return S.REDUCE_WAIT_PHOTO
    await a_secure_store(ctx, "reduce_img", img)
    img = None; gc.collect()
    await svc_prompt(update, ctx, t("enter_kb", ctx))
    return S.REDUCE_WAIT_KB

async def reduce_wait_kb(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    val = parse_size_kb(update.message.text or "")
    if not val:
        await svc_prompt(update, ctx, t("enter_kb", ctx)); return S.REDUCE_WAIT_KB
    ctx.user_data["target_kb"] = val
    await svc_prompt(update, ctx, t("format_choose", ctx), format_kb())
    return S.REDUCE_WAIT_FORMAT

async def reduce_wait_format(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query; await q.answer()
    fmt = q.data.replace("fmt_", "")
    img = await a_secure_load(ctx, "reduce_img")
    if not img:
        await svc_prompt(update, ctx, t("error", ctx))
        cleanup_session(ctx); await send_main_menu(update, ctx)
        return S.SELECT_ACTION
    dims = img.size                                    # v6.5: pehle capture
    target_kb = ctx.user_data.get("target_kb", 100)
    proc = await svc_prompt(update, ctx, t("processing", ctx))
    try:
        buf = await asyncio.wait_for(
            asyncio.to_thread(compress_to_kb, img, target_kb, fmt),
            timeout=PROCESSING_TIMEOUT)
        data = buf.getvalue(); buf.close()
        img = None; gc.collect()
        await deliver_result(update, ctx, q.message, data,
                             f"output.{fmt.lower()}", dims, "compress_done")
        data = None; gc.collect()
    except asyncio.TimeoutError:
        img = None; gc.collect()
        await proc.edit_text(t("timeout_err", ctx))
    except Exception as e:
        img = None; gc.collect()
        logger.error(f"reduce_wait_format: {e}", exc_info=True)
        await proc.edit_text(t("error", ctx))
    finally:
        cleanup_session(ctx)
        await send_main_menu(update, ctx)
    return S.SELECT_ACTION

# ─────────────────────────────────────────────────────────────────────
# SIGNATURE FLOW
# ─────────────────────────────────────────────────────────────────────
async def sig_wait_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    uid = update.effective_user.id
    _known_real_ids.add(uid)
    if not check_rate_limit(uid):
        await safe_reply(update, t("rate_limit", ctx)); return S.SIG_WAIT_PHOTO
    img = await get_image(update)
    if img == "too_large":
        await safe_reply(update, t("file_too_large", ctx)); return S.SIG_WAIT_PHOTO
    if img in ("invalid", None):
        await safe_reply(update, t("invalid_file" if img == "invalid" else "no_photo", ctx))
        return S.SIG_WAIT_PHOTO
    if not await asyncio.to_thread(_sig_bg_is_white, img):    # v6.5: thread mein
        await safe_reply(update, t("sig_bg_warn", ctx))
    proc = await svc_prompt(update, ctx, t("processing", ctx))
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(extract_signature, img),
            timeout=PROCESSING_TIMEOUT)
        img = None; gc.collect()

        # v6.5: empty-signature detection (alpha channel mean)
        alpha_mean = await asyncio.to_thread(
            lambda r: float(np.asarray(r)[..., 3].mean()), result)
        if alpha_mean < 3.0:
            await safe_reply(update, t("sig_empty_warn", ctx))

        preview = await asyncio.to_thread(create_preview, result)
        await a_secure_store(ctx, "sig_result", result)
        result = None; gc.collect()
        await svc_prompt_photo(update, ctx, preview, t("preview", ctx),
                               confirm_kb("sig_ok", "sig_retry", ctx))
        return S.SIG_PREVIEW
    except asyncio.TimeoutError:
        img = None; gc.collect()
        await proc.edit_text(t("timeout_err", ctx))
        cleanup_session(ctx); await send_main_menu(update, ctx)
        return S.SELECT_ACTION
    except Exception as e:
        img = None; gc.collect()
        logger.error(f"sig_wait_photo: {e}", exc_info=True)
        await proc.edit_text(t("error", ctx))
        cleanup_session(ctx); await send_main_menu(update, ctx)
        return S.SELECT_ACTION

async def sig_preview(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query; await q.answer()
    if q.data == "sig_ok":
        try:
            await q.edit_message_caption(caption=t("format_choose", ctx),
                                         reply_markup=sig_format_kb(),
                                         parse_mode="Markdown")
        except BadRequest:
            await svc_prompt(update, ctx, t("format_choose", ctx), sig_format_kb())
        return S.SIG_WAIT_FORMAT
    elif q.data == "sig_retry":
        secure_wipe_all(ctx, ["sig_result"])
        await svc_prompt(update, ctx, t("send_photo", ctx))
        return S.SIG_WAIT_PHOTO
    return S.SIG_PREVIEW

async def sig_wait_format(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query; await q.answer()
    fmt = q.data.replace("fmt_", "")
    result = await a_secure_load(ctx, "sig_result")
    if not result:
        await svc_prompt(update, ctx, t("error", ctx))
        cleanup_session(ctx); await send_main_menu(update, ctx)
        return S.SELECT_ACTION
    dims = result.size
    try:
        def _export():
            b = io.BytesIO()
            if fmt == "PNG":
                result.save(b, format="PNG")
            elif fmt == "JPEG":
                flatten_on_white(result).save(b, format="JPEG", quality=MAX_QUALITY,
                                              dpi=(DPI_DEFAULT, DPI_DEFAULT), optimize=True)
            else:
                flatten_on_white(result).save(b, format="PDF", resolution=DPI_DEFAULT)
            return b.getvalue()
        data = await asyncio.to_thread(_export)
        result = None; gc.collect()
        await deliver_result(update, ctx, q.message, data,
                             f"signature.{fmt.lower()}", dims, "sig_done")
        data = None; gc.collect()
    except Exception as e:
        result = None; gc.collect()
        logger.error(f"sig_wait_format: {e}", exc_info=True)
        await safe_reply(update, t("error", ctx))
    finally:
        cleanup_session(ctx)
        await send_main_menu(update, ctx)
    return S.SELECT_ACTION

# ─────────────────────────────────────────────────────────────────────
# 🎯 EXACT SIZE MATCH FLOW
# ─────────────────────────────────────────────────────────────────────
async def size_wait_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    uid = update.effective_user.id
    _known_real_ids.add(uid)
    if not check_rate_limit(uid):
        await safe_reply(update, t("rate_limit", ctx)); return S.SIZE_WAIT_PHOTO
    img = await get_image(update)
    if img == "too_large":
        await safe_reply(update, t("file_too_large", ctx)); return S.SIZE_WAIT_PHOTO
    if img in ("invalid", None):
        await safe_reply(update, t("invalid_file" if img == "invalid" else "no_photo", ctx))
        return S.SIZE_WAIT_PHOTO
    await a_secure_store(ctx, "size_img", img)   # enhancement NAHI — pixels untouched
    img = None; gc.collect()
    await svc_prompt(update, ctx, t("size_enter_kb", ctx))
    return S.SIZE_WAIT_KB

async def size_wait_kb(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    val = parse_size_kb(update.message.text or "")
    if not val:
        await svc_prompt(update, ctx, t("size_enter_kb", ctx)); return S.SIZE_WAIT_KB
    ctx.user_data["target_kb"] = val
    await svc_prompt(update, ctx, t("size_fmt_choose", ctx), size_fmt_kb())
    return S.SIZE_WAIT_FORMAT

async def size_wait_format(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    q = update.callback_query; await q.answer()
    fmt = q.data.replace("fmt_", "")
    img = await a_secure_load(ctx, "size_img")
    if not img:
        await svc_prompt(update, ctx, t("error", ctx))
        cleanup_session(ctx); await send_main_menu(update, ctx)
        return S.SELECT_ACTION
    target_kb = ctx.user_data.get("target_kb", 100)
    orig_dims = img.size
    proc = await svc_prompt(update, ctx, t("processing", ctx))
    try:
        buf, meta = await asyncio.wait_for(
            asyncio.to_thread(match_file_size_kb, img, target_kb, fmt),
            timeout=PROCESSING_TIMEOUT)
        data = buf.getvalue(); buf.close()
        img = None; gc.collect()

        info = t("size_info", ctx).format(
            w=orig_dims[0], h=orig_dims[1],
            size=format_size(len(data)), target=f"{target_kb}KB",
            q=meta.get("quality", "?"))
        warn_keys = {"min": "size_min_warn", "max": "size_max_warn"}
        warn_txt = ""
        if meta.get("warn") in warn_keys:
            warn_txt = "\n" + t(warn_keys[meta["warn"]], ctx)
        elif meta.get("warn") == "quality":
            warn_txt = "\n" + t("size_q_warn", ctx).format(q=meta.get("quality", "?"))
        caption = f"{t('size_done', ctx)}\n{info}{warn_txt}\n\n{t('reminder', ctx)}"

        _store_history(ctx, data, f"photo_{target_kb}kb.{fmt.lower()}")
        await asyncio.to_thread(bump_op_count, update.effective_user.id)
        await q.message.reply_document(
            document=io.BytesIO(data),
            filename=f"photo_{target_kb}kb.{fmt.lower()}",
            caption=caption, parse_mode="Markdown")
        data = None; gc.collect()
    except asyncio.TimeoutError:
        img = None; gc.collect()
        await proc.edit_text(t("timeout_err", ctx))
    except Exception as e:
        img = None; gc.collect()
        logger.error(f"size_wait_format: {e}", exc_info=True)
        await proc.edit_text(t("error", ctx))
    finally:
        cleanup_session(ctx)
        await send_main_menu(update, ctx)
    return S.SELECT_ACTION

# ─────────────────────────────────────────────────────────────────────
# 🖨 PRINT SHEET FLOW
# ─────────────────────────────────────────────────────────────────────
async def sheet_wait_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int:
    uid = update.effective_user.id
    _known_real_ids.add(uid)
    if not check_rate_limit(uid):
        await safe_reply(update, t("rate_limit", ctx)); return S.SHEET_WAIT_PHOTO
    img = await get_image(update)
    if img == "too_large":
        await safe_reply(update, t("file_too_large", ctx)); return S.SHEET_WAIT_PHOTO
    if img in ("invalid", None):
        await safe_reply(update, t("invalid_file" if img == "invalid" else "no_photo", ctx))
        return S.SHEET_WAIT_PHOTO
    fb, _fw, img = await asyncio.to_thread(_preflight, img, ctx, False)
    await safe_reply(update, fb)
    proc = await svc_prompt(update, ctx, t("processing", ctx))
    try:
        sheet = await asyncio.wait_for(
            asyncio.to_thread(make_photo_sheet, img),
            timeout=PROCESSING_TIMEOUT)
        img = None; gc.collect()
        jpg_data, pdf_data = await asyncio.to_thread(_sheet_encode, sheet)
        sheet = None; gc.collect()

        _store_history(ctx, jpg_data, "print_sheet_4x6.jpg")
        await asyncio.to_thread(bump_op_count, update.effective_user.id)
        await update.message.reply_document(
            document=io.BytesIO(jpg_data), filename="print_sheet_4x6.jpg",
            caption=t("sheet_done", ctx), parse_mode="Markdown")
        await update.message.reply_document(
            document=io.BytesIO(pdf_data), filename="print_sheet_4x6.pdf",
            caption="📄 PDF version — direct print ke liye")
        jpg_data = pdf_data = None; gc.collect()
    except asyncio.TimeoutError:
        img = None; gc.collect()
        await proc.edit_text(t("timeout_err", ctx))
    except Exception as e:
        img = None; gc.collect()
        logger.error(f"sheet_wait_photo: {e}", exc_info=True)
        await proc.edit_text(t("error", ctx))
    finally:
        cleanup_session(ctx)
        await send_main_menu(update, ctx)
    return S.SELECT_ACTION

# ─────────────────────────────────────────────────────────────────────
# FALLBACKS & ERROR HANDLER
# ════════════════════════════════════════════════════════════════════
# v6.4 fix retained: fallback STATE PRESERVE karta hai (return None).
# Lock/images kabhi orphan nahi hote, user flow kabhi hijack nahi hota.
# ════════════════════════════════════════════════════════════════════
async def conversation_fallback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    if update.callback_query:
        await update.callback_query.answer(t("unexpected", ctx), show_alert=True)
    elif update.message:
        await safe_reply(update, t("unexpected", ctx))
    return None                        # state UNCHANGED — lock/images safe

async def global_fallback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Conversation ke bahar girne wala KUCH BHI — callbacks included."""
    if update.effective_user:
        _known_real_ids.add(update.effective_user.id)
    if update.callback_query:
        await update.callback_query.answer()
    await send_main_menu(update, ctx)

async def error_handler(update: object, ctx: ContextTypes.DEFAULT_TYPE):
    err = ctx.error
    logger.error(f"Unhandled error: {err}", exc_info=err)
    if isinstance(err, (TimedOut, NetworkError, RetryAfter)):
        return
    if update and hasattr(update, "effective_chat") and update.effective_chat:
        try:
            await ctx.bot.send_message(update.effective_chat.id,
                                       "⚠️ Unexpected error. Please /start again.")
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────
# LIFECYCLE
# ─────────────────────────────────────────────────────────────────────
async def post_init(application: Application):
    public_commands = [
        BotCommand("start",    "Main menu"),
        BotCommand("help",     "How to use"),
        BotCommand("privacy",  "Privacy policy"),
        BotCommand("cancel",   "Cancel current operation"),
        BotCommand("history",  "Resend last result"),
        BotCommand("mystats",  "Your usage stats"),
        BotCommand("hinglish", "Toggle Hinglish/English"),
        BotCommand("strict",   "Toggle pad/crop mode"),
        BotCommand("dpi",      "Set output DPI"),
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
                admin_commands, scope={"type": "chat", "chat_id": admin_id})
        except Exception:
            pass
    logger.info("Bot commands registered.")
    warm_up_model()

async def post_shutdown(application: Application):
    _wipe_session_key()
    logger.info("Session key wiped. Goodbye.")

# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    token = os.environ.get("BOT_TOKEN")
    if not token:
        logger.error("BOT_TOKEN not set!"); sys.exit(1)
    if not HEIC_OK:
        logger.warning("pillow-heif missing — iPhone HEIC rejected")
    if not MP_OK:
        logger.warning("mediapipe missing — BG change & face-crop degraded")
    if not WAITRESS_OK:
        logger.warning("waitress missing — Flask dev server chalega (pip install waitress)")

    init_db()
    logger.info(f"Database ready: {DB_PATH}")
    threading.Thread(target=run_flask, daemon=True).start()
    logger.info(f"Health server on port {os.environ.get('PORT', 8080)} "
                f"({'waitress' if WAITRESS_OK else 'flask-dev'})")

    application = (Application.builder()
                   .token(token)
                   .post_init(post_init)
                   .post_shutdown(post_shutdown)
                   .build())

    # NOTE: /start aur /cancel jaan-boojh ke external list mein NAHI —
    # conversation ke entry_points/fallbacks se route hote hain.
    # /help, /history etc. external hain aur NON-DESTRUCTIVE — ye
    # conversation se PEHLE add hain, isliye mid-flow /help session
    # ko touch nahi karta (user ka chalu operation preserve hota hai).
    for cmd, fn in [
        ("help",        cmd_help),
        ("privacy",     cmd_privacy),
        ("hinglish",    cmd_hinglish),
        ("strict",      cmd_strict),
        ("dpi",         cmd_dpi),
        ("history",     cmd_history),
        ("mystats",     cmd_mystats),
        ("admin",       cmd_admin),
        ("broadcast",   cmd_broadcast),
        ("listpresets", cmd_listpresets),
        ("addpreset",   cmd_addpreset),
        ("editpreset",  cmd_editpreset),
        ("delpreset",   cmd_delpreset),
    ]:
        application.add_handler(CommandHandler(cmd, fn))

    photo_filter = filters.PHOTO | filters.Document.IMAGE

    conv = ConversationHandler(
        entry_points=[
            CommandHandler("start", cmd_start),
            # Menu actions ENTRY POINTS — conversation dead ho tab bhi
            # click → re-enter → turant kaam
            CallbackQueryHandler(select_action, pattern=MENU_ACTION_PATTERN),
        ],
        states={
            S.SELECT_ACTION:        [CallbackQueryHandler(select_action)],
            S.BG_WAIT_PHOTO:        [MessageHandler(photo_filter, bg_wait_photo)],
            S.BG_WAIT_COLOR:        [
                CallbackQueryHandler(bg_wait_color, pattern=r"^col_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, bg_wait_color),
            ],
            S.BG_PREVIEW:           [CallbackQueryHandler(bg_preview,     pattern=r"^bg_")],
            S.BG_WAIT_FORMAT:       [CallbackQueryHandler(bg_wait_format, pattern=r"^fmt_")],
            S.RESIZE_MODE:          [CallbackQueryHandler(resize_mode,    pattern=r"^resize_")],
            S.CUSTOM_WAIT_PHOTO:    [MessageHandler(photo_filter, custom_wait_photo)],
            S.CUSTOM_SELECT_PRESET: [CallbackQueryHandler(custom_select_preset, pattern=r"^preset_")],
            S.CUSTOM_WAIT_DIMS:     [MessageHandler(filters.TEXT & ~filters.COMMAND,
                                                    custom_wait_dims)],
            S.CUSTOM_PREVIEW:       [CallbackQueryHandler(custom_preview, pattern=r"^resize_")],
            S.CUSTOM_SIZE_OPT:      [CallbackQueryHandler(custom_size_opt, pattern=r"^sizeopt_")],
            S.CUSTOM_WAIT_KB:       [MessageHandler(filters.TEXT & ~filters.COMMAND,
                                                    custom_wait_kb)],
            S.CUSTOM_WAIT_FORMAT:   [CallbackQueryHandler(custom_wait_format, pattern=r"^fmt_")],
            S.REDUCE_WAIT_PHOTO:    [MessageHandler(photo_filter, reduce_wait_photo)],
            S.REDUCE_WAIT_KB:       [MessageHandler(filters.TEXT & ~filters.COMMAND,
                                                    reduce_wait_kb)],
            S.REDUCE_WAIT_FORMAT:   [CallbackQueryHandler(reduce_wait_format, pattern=r"^fmt_")],
            S.SIG_WAIT_PHOTO:       [MessageHandler(photo_filter, sig_wait_photo)],
            S.SIG_PREVIEW:          [CallbackQueryHandler(sig_preview,   pattern=r"^sig_")],
            S.SIG_WAIT_FORMAT:      [CallbackQueryHandler(sig_wait_format, pattern=r"^fmt_")],
            S.SHEET_WAIT_PHOTO:     [MessageHandler(photo_filter, sheet_wait_photo)],
            S.SIZE_WAIT_PHOTO:      [MessageHandler(photo_filter, size_wait_photo)],
            S.SIZE_WAIT_KB:         [MessageHandler(filters.TEXT & ~filters.COMMAND,
                                                    size_wait_kb)],
            S.SIZE_WAIT_FORMAT:     [CallbackQueryHandler(size_wait_format, pattern=r"^fmt_")],
        },
        fallbacks=[
            CommandHandler("cancel",  cmd_cancel),
            CommandHandler("start",   cmd_start),
            MessageHandler(filters.ALL, conversation_fallback),
        ],
        allow_reentry=True,
    )

    application.add_handler(conv)
    # Callback catch-all — conversation ke bahar girne wala har click
    # answer hota hai (spinner kabhi atka nahi)
    application.add_handler(CallbackQueryHandler(global_fallback))
    application.add_handler(MessageHandler(filters.ALL, global_fallback))
    application.add_error_handler(error_handler)

    logger.info(f"🚀 {VERSION} starting — fully audited production build")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
