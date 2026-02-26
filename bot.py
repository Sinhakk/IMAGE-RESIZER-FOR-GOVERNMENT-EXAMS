#!/usr/bin/env python3
"""
Image Utility Telegram Bot — Government Exam Photo Helper
Version: v3.0-render-ready
Features: Background Change, Resize, Signature Extract
Deploy: Render Free Tier (Polling mode + Flask health check)
"""

import os
import io
import re
import cv2
import numpy as np
import logging
import time
import asyncio
import gc
import signal
import sys
import threading
from enum import Enum
from collections import defaultdict
from threading import Lock
from PIL import Image, ImageColor, ImageOps, ImageFilter, ImageEnhance, PngImagePlugin
from flask import Flask, jsonify
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)
from telegram.error import BadRequest, TimedOut, NetworkError
import mediapipe as mp

# ─────────────────────────────────────────────────────────────────────
# FLASK HEALTH CHECK (Required for Render to keep service alive)
# ─────────────────────────────────────────────────────────────────────
flask_app = Flask(__name__)

@flask_app.route("/")
def home():
    return jsonify({"status": "running", "bot": "Image Utility Bot v3.0"})

@flask_app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    flask_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

# ─────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────
VERSION = "v3.0-render-ready"
DPI = 300
MAX_QUALITY = 95
MIN_QUALITY = 10
PREVIEW_MAX_SIZE = (400, 400)
BLUR_THRESHOLD = 100
MAX_IMAGE_DIMENSION = 4000
PROCESSING_TIMEOUT = 25
RATE_LIMIT_REQUESTS = 5
RATE_LIMIT_PERIOD = 60

# ─────────────────────────────────────────────────────────────────────
# MEDIAPIPE SINGLETON (thread-safe)
# ─────────────────────────────────────────────────────────────────────
_mp_model = None
_mp_lock = Lock()

def get_segmentation_model():
    global _mp_model
    with _mp_lock:
        if _mp_model is None:
            _mp_model = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    return _mp_model

# ─────────────────────────────────────────────────────────────────────
# LANGUAGE SUPPORT
# ─────────────────────────────────────────────────────────────────────
LANG_HINGLISH = {
    "start": "Namaste! Kya karna chahenge?",
    "bg_change": "🖼 Background Change",
    "resize": "📐 Resize Image",
    "signature": "✍️ Signature Extract",
    "send_photo": "📸 Photo bhejo (JPEG/PNG)",
    "processing": "⏳ Thoda intezaar karein...",
    "preview": "Preview dekh lo. Theek hai?",
    "looks_ok": "✅ Theek hai",
    "change_settings": "🔄 Badalna hai",
    "format_choose": "Format chuno:",
    "size_target": "File size kitna chahiye? (KB mein, jaise 50)",
    "dimensions": "Dimensions do (jaise: 300x200 ya 3.5x4.5cm)",
    "color": "Rang batao (naam ya hex, jaise: white, #FF0000)",
    "error": "❌ Kuch gadbad hui. Phir se try karo.",
    "cancel": "✋ Cancel kar diya.",
    "blur_warn": "⚠️ Photo thodi blur lag rahi hai. Better photo use karo.",
    "signature_bg_warn": "⚠️ Signature background safed nahi hai. Safed paper use karo.",
    "reminder": "⚠️ Upload se pehle preview zaroor check karo. Ye tool sirf madad ke liye hai.",
    "bg_warning": "⚠️ Background mein kami reh sakti hai. Preview dhyan se dekhein.",
    "main_menu": "🏠 Main Menu — kya karna hai?",
    "unexpected": "⚠️ Samajh nahi aaya. Niche ke options use karo.",
    "size_too_large": "⚠️ Photo bahut badi hai. Chhota kar rahe hain...",
    "rate_limit": "⏳ Bahut zyada requests. Thodi der ruk kar try karein.",
    "timeout_error": "⏱ Processing mein time lag raha hai. Kam resolution ki photo use karein.",
    "face_small_warning": "⚠️ Face chhota hai. Better photo use karein.",
    "face_offcenter_warning": "⚠️ Face centre mein nahi hai. Framing theek karein.",
    "background_presets": "Background color chuno:",
    "history": "Last processed image dubara bhejna hai?",
    "no_history": "Koi previous image nahi mili.",
    "strict_on": "✅ Strict mode ON — aspect ratio strictly maintain hoga.",
    "strict_off": "❌ Strict mode OFF.",
    "lang_hinglish": "✅ Hinglish mode ON!",
    "lang_english": "✅ English mode ON!",
    "dpi_set": "DPI set ho gaya: ",
    "dpi_usage": "Usage: /dpi 72 ya /dpi 300",
    "size_option": "Size reduce kaise karna hai?",
    "size_by_kb": "📦 KB target set karo",
    "size_by_dims": "📐 Dimensions se resize karo",
    "enter_kb": "Target size KB mein batao (jaise: 50):",
    "done": "✅ Ho gaya!",
    "reduce_send_photo": "📸 Photo bhejo jiska size reduce karna hai:",
}

LANG_EN = {
    "start": "Welcome! What would you like to do?",
    "bg_change": "🖼 Background Change",
    "resize": "📐 Resize Image",
    "signature": "✍️ Signature Extract",
    "send_photo": "📸 Send photo (JPEG/PNG)",
    "processing": "⏳ Processing... please wait.",
    "preview": "Preview the result. Looks OK?",
    "looks_ok": "✅ Looks OK",
    "change_settings": "🔄 Change Settings",
    "format_choose": "Choose format:",
    "size_target": "Enter desired file size in KB (e.g. 50):",
    "dimensions": "Provide dimensions (e.g., 300x200 or 3.5x4.5cm)",
    "color": "Enter color (name or hex, e.g. white, #FF0000):",
    "error": "❌ Something went wrong. Please try again.",
    "cancel": "✋ Cancelled.",
    "blur_warn": "⚠️ Photo appears blurry. Use a clearer image.",
    "signature_bg_warn": "⚠️ Signature background is not white. Use plain white paper.",
    "reminder": "⚠️ Double-check output before uploading. This tool is for assistance only.",
    "bg_warning": "⚠️ Background change may have imperfections. Check preview carefully.",
    "main_menu": "🏠 Main Menu — What would you like to do?",
    "unexpected": "⚠️ Unexpected input. Please use the buttons below.",
    "size_too_large": "⚠️ Image is too large. Downsizing...",
    "rate_limit": "⏳ Too many requests. Please wait a moment.",
    "timeout_error": "⏱ Processing timeout. Try a lower resolution image.",
    "face_small_warning": "⚠️ Face is too small. Use a better photo.",
    "face_offcenter_warning": "⚠️ Face is not centered. Improve framing.",
    "background_presets": "Choose background color:",
    "history": "Resend last processed image?",
    "no_history": "No previous image found.",
    "strict_on": "✅ Strict mode ON — aspect ratio strictly enforced.",
    "strict_off": "❌ Strict mode OFF.",
    "lang_hinglish": "✅ Hinglish mode ON!",
    "lang_english": "✅ English mode ON!",
    "dpi_set": "DPI set to: ",
    "dpi_usage": "Usage: /dpi 72 or /dpi 300",
    "size_option": "How do you want to reduce size?",
    "size_by_kb": "📦 Set KB target",
    "size_by_dims": "📐 Resize by dimensions",
    "enter_kb": "Enter target size in KB (e.g. 50):",
    "done": "✅ Done!",
    "reduce_send_photo": "📸 Send the photo you want to reduce:",
}

def _(key: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    if context.user_data.get("hinglish", False):
        return LANG_HINGLISH.get(key, LANG_EN.get(key, key))
    return LANG_EN.get(key, key)

# ─────────────────────────────────────────────────────────────────────
# CONVERSATION STATES
# ─────────────────────────────────────────────────────────────────────
class States(Enum):
    SELECT_ACTION        = 0
    BG_WAIT_PHOTO        = 1
    BG_WAIT_COLOR        = 2
    BG_PREVIEW           = 3
    BG_WAIT_FORMAT       = 19
    RESIZE_MODE          = 4
    CUSTOM_WAIT_PHOTO    = 5
    CUSTOM_WAIT_DIMS     = 6
    CUSTOM_WAIT_SIZE_OPT = 7
    CUSTOM_WAIT_KB       = 8
    CUSTOM_WAIT_FORMAT   = 20
    CUSTOM_PREVIEW       = 9
    REDUCE_WAIT_PHOTO    = 10
    REDUCE_WAIT_KB       = 11
    REDUCE_WAIT_FORMAT   = 21
    REDUCE_PREVIEW       = 12
    SIG_WAIT_PHOTO       = 13
    SIG_WAIT_HEIGHT      = 14
    SIG_WAIT_KB          = 15
    SIG_WAIT_FORMAT      = 22
    SIG_PREVIEW          = 16

# ─────────────────────────────────────────────────────────────────────
# RATE LIMITING
# ─────────────────────────────────────────────────────────────────────
_rate_data: dict = defaultdict(list)

def check_rate_limit(user_id: int) -> bool:
    now = time.time()
    _rate_data[user_id] = [t for t in _rate_data[user_id] if now - t < RATE_LIMIT_PERIOD]
    if len(_rate_data[user_id]) >= RATE_LIMIT_REQUESTS:
        return False
    _rate_data[user_id].append(now)
    return True

# ─────────────────────────────────────────────────────────────────────
# PROCESSING LOCK
# ─────────────────────────────────────────────────────────────────────
def acquire_lock(context: ContextTypes.DEFAULT_TYPE) -> bool:
    if context.user_data.get("processing"):
        return False
    context.user_data["processing"] = True
    return True

def release_lock(context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("processing", None)

# ─────────────────────────────────────────────────────────────────────
# IMAGE UTILITIES
# ─────────────────────────────────────────────────────────────────────
def fix_orientation(img: Image.Image) -> Image.Image:
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img

def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img

def check_image_size(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img

def detect_blur(img: Image.Image) -> float:
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def analyze_face(img: Image.Image):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    if len(faces) == 0:
        return None, 0
    x, y, w, h = faces[0]
    iw, ih = img.size
    if (w * h) < (iw * ih * 0.04):
        return "face_small_warning", len(faces)
    cx = x + w / 2
    if abs(cx - iw / 2) > iw * 0.25:
        return "face_offcenter_warning", len(faces)
    return None, len(faces)

def create_preview(img: Image.Image) -> io.BytesIO:
    img.thumbnail(PREVIEW_MAX_SIZE, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=80)
    buf.seek(0)
    return buf

def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"

def parse_dimensions(text: str):
    text = text.strip().lower()
    m = re.match(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*cm", text)
    if m:
        w = int(float(m.group(1)) / 2.54 * DPI)
        h = int(float(m.group(2)) / 2.54 * DPI)
        return w, h
    m = re.match(r"(\d+)\s*x\s*(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None

def parse_color(text: str) -> tuple:
    try:
        return ImageColor.getrgb(text.strip())[:3]
    except Exception:
        return (255, 255, 255)

def compress_to_kb(img: Image.Image, target_kb: int, fmt: str = "JPEG") -> io.BytesIO:
    buf = io.BytesIO()
    quality = MAX_QUALITY
    fmt_upper = fmt.upper()
    while quality >= MIN_QUALITY:
        buf.seek(0)
        buf.truncate()
        if fmt_upper == "JPEG":
            img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
        elif fmt_upper == "PNG":
            img.save(buf, format="PNG", optimize=True)
        elif fmt_upper == "PDF":
            img.convert("RGB").save(buf, format="PDF")
        if buf.tell() <= target_kb * 1024:
            break
        quality -= 5
    buf.seek(0)
    return buf

def save_image(img: Image.Image, fmt: str, dpi_val: int) -> io.BytesIO:
    buf = io.BytesIO()
    fmt_upper = fmt.upper()
    if fmt_upper == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=MAX_QUALITY, dpi=(dpi_val, dpi_val))
    elif fmt_upper == "PNG":
        info = PngImagePlugin.PngInfo()
        img.save(buf, format="PNG", pnginfo=info)
    elif fmt_upper == "PDF":
        img.convert("RGB").save(buf, format="PDF", resolution=dpi_val)
    buf.seek(0)
    return buf

# ─────────────────────────────────────────────────────────────────────
# CORE PROCESSING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────
def person_segmentation_replace(img: Image.Image, color_text: str) -> Image.Image:
    rgb = np.array(img.convert("RGB"))
    model = get_segmentation_model()
    with _mp_lock:
        result = model.process(rgb)
    mask = result.segmentation_mask
    mask_bin = (mask > 0.6).astype(np.uint8) * 255
    mask_bin = cv2.GaussianBlur(mask_bin, (21, 21), 0)
    mask_f = mask_bin.astype(np.float32) / 255.0
    color = parse_color(color_text)
    bg = np.full_like(rgb, color, dtype=np.uint8)
    out = (rgb.astype(np.float32) * mask_f[..., None] +
           bg.astype(np.float32) * (1 - mask_f[..., None])).astype(np.uint8)
    return Image.fromarray(out)

def extract_signature(img: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    rgba = np.zeros((*gray.shape, 4), dtype=np.uint8)
    rgba[..., 3] = thresh
    return Image.fromarray(rgba, "RGBA")

# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
async def safe_edit_or_send(update: Update, text: str, reply_markup=None):
    if update.callback_query:
        query = update.callback_query
        try:
            if query.message.text:
                await query.edit_message_text(text=text, reply_markup=reply_markup)
            elif query.message.caption:
                await query.edit_message_caption(caption=text, reply_markup=reply_markup)
            else:
                await query.message.reply_text(text=text, reply_markup=reply_markup)
        except BadRequest as e:
            logger.warning(f"Edit failed: {e}")
            await query.message.reply_text(text=text, reply_markup=reply_markup)
    elif update.message:
        await update.message.reply_text(text=text, reply_markup=reply_markup)

async def get_image_from_update(update: Update) -> Image.Image | None:
    photo = update.message.photo[-1] if update.message.photo else None
    doc = update.message.document if update.message.document else None
    file_obj = photo or doc
    if not file_obj:
        return None
    tg_file = await file_obj.get_file()
    buf = io.BytesIO()
    await tg_file.download_to_memory(buf)
    buf.seek(0)
    img = Image.open(buf)
    img = fix_orientation(img)
    img = ensure_rgb(img)
    img = check_image_size(img)
    return img

def cleanup(context: ContextTypes.DEFAULT_TYPE, keys: list):
    for k in keys:
        context.user_data.pop(k, None)
    gc.collect()

def log_state(user_id, state, action):
    logger.info(f"USER {user_id} | STATE {state} | ACTION {action}")

# ─────────────────────────────────────────────────────────────────────
# KEYBOARDS
# ─────────────────────────────────────────────────────────────────────
def main_menu_kb(context):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(_("bg_change", context), callback_data="bg_change")],
        [InlineKeyboardButton(_("resize", context), callback_data="resize")],
        [InlineKeyboardButton(_("signature", context), callback_data="signature")],
    ])

def bg_color_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚪ White", callback_data="color_white"),
         InlineKeyboardButton("🟡 Off-white", callback_data="color_offwhite")],
        [InlineKeyboardButton("🔵 Light Blue", callback_data="color_lightblue"),
         InlineKeyboardButton("⬜ Light Grey", callback_data="color_lightgrey")],
        [InlineKeyboardButton("🎨 Custom", callback_data="color_custom")],
    ])

def confirm_kb(ok_cb, change_cb, context):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(_("looks_ok", context), callback_data=ok_cb),
         InlineKeyboardButton(_("change_settings", context), callback_data=change_cb)],
    ])

def format_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("JPEG", callback_data="fmt_JPEG"),
         InlineKeyboardButton("PNG", callback_data="fmt_PNG"),
         InlineKeyboardButton("PDF", callback_data="fmt_PDF")],
    ])

def resize_mode_kb(context):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📐 Custom Dimensions", callback_data="resize_custom")],
        [InlineKeyboardButton("📦 Reduce File Size", callback_data="resize_reduce")],
    ])

def size_option_kb(context):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(_("size_by_kb", context), callback_data="size_kb")],
        [InlineKeyboardButton(_("size_by_dims", context), callback_data="size_dims")],
    ])

# ─────────────────────────────────────────────────────────────────────
# SEND MAIN MENU
# ─────────────────────────────────────────────────────────────────────
async def send_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = _("main_menu", context)
    kb = main_menu_kb(context)
    if update.callback_query:
        await safe_edit_or_send(update, text, kb)
    elif update.message:
        await update.message.reply_text(text, reply_markup=kb)

# ─────────────────────────────────────────────────────────────────────
# COMMAND HANDLERS
# ─────────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if context.user_data.get("processing"):
        await update.message.reply_text(_("processing", context))
        return ConversationHandler.END
    settings = {k: context.user_data[k] for k in ("hinglish", "strict", "dpi", "last_image") if k in context.user_data}
    context.user_data.clear()
    context.user_data.update(settings)
    log_state(update.effective_user.id, "START", "/start")
    await send_main_menu(update, context)
    return States.SELECT_ACTION

async def cmd_hinglish(update: Update, context: ContextTypes.DEFAULT_TYPE):
    current = context.user_data.get("hinglish", False)
    context.user_data["hinglish"] = not current
    key = "lang_hinglish" if not current else "lang_english"
    await update.message.reply_text(_(key, context))
    await send_main_menu(update, context)

async def cmd_strict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    current = context.user_data.get("strict", False)
    context.user_data["strict"] = not current
    key = "strict_on" if not current else "strict_off"
    await update.message.reply_text(_(key, context))

async def cmd_dpi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if args and args[0] in ("72", "96", "150", "300"):
        context.user_data["dpi"] = int(args[0])
        await update.message.reply_text(_("dpi_set", context) + args[0])
    else:
        await update.message.reply_text(_("dpi_usage", context))

async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    last = context.user_data.get("last_image")
    if last:
        buf = io.BytesIO(last)
        buf.seek(0)
        await update.message.reply_photo(photo=buf, caption=_("history", context))
    else:
        await update.message.reply_text(_("no_history", context))

async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    release_lock(context)
    await update.message.reply_text(_("cancel", context))
    await send_main_menu(update, context)
    return ConversationHandler.END

# ─────────────────────────────────────────────────────────────────────
# SELECT ACTION
# ─────────────────────────────────────────────────────────────────────
async def select_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data
    uid = update.effective_user.id

    if not check_rate_limit(uid):
        await safe_edit_or_send(update, _("rate_limit", context))
        return ConversationHandler.END

    log_state(uid, "SELECT_ACTION", data)

    if data == "bg_change":
        await safe_edit_or_send(update, _("send_photo", context))
        return States.BG_WAIT_PHOTO
    elif data == "resize":
        await safe_edit_or_send(update, _("resize", context), resize_mode_kb(context))
        return States.RESIZE_MODE
    elif data == "signature":
        if not acquire_lock(context):
            await safe_edit_or_send(update, _("processing", context))
            return ConversationHandler.END
        await safe_edit_or_send(update, _("send_photo", context))
        return States.SIG_WAIT_PHOTO

    return States.SELECT_ACTION

# ─────────────────────────────────────────────────────────────────────
# BACKGROUND CHANGE FLOW
# ─────────────────────────────────────────────────────────────────────
async def bg_wait_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    uid = update.effective_user.id
    if not check_rate_limit(uid):
        await update.message.reply_text(_("rate_limit", context))
        return ConversationHandler.END
    if not acquire_lock(context):
        await update.message.reply_text(_("processing", context))
        return ConversationHandler.END

    try:
        img = await get_image_from_update(update)
        if not img:
            await update.message.reply_text(_("send_photo", context))
            release_lock(context)
            return States.BG_WAIT_PHOTO

        context.user_data["bg_img"] = img

        if detect_blur(img) < BLUR_THRESHOLD:
            await update.message.reply_text(_("blur_warn", context))
        warn, _ = analyze_face(img)
        if warn:
            await update.message.reply_text(_(warn, context))

        await update.message.reply_text(_("background_presets", context), reply_markup=bg_color_kb())
        return States.BG_WAIT_COLOR

    except Exception as e:
        logger.error(f"bg_wait_photo uid={uid}: {e}")
        await update.message.reply_text(_("error", context))
        release_lock(context)
        await send_main_menu(update, context)
        return ConversationHandler.END

async def bg_wait_color(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    uid = update.effective_user.id
    color_text = None

    if update.callback_query:
        await update.callback_query.answer()
        data = update.callback_query.data
        color_map = {
            "color_white": "white",
            "color_offwhite": "#f8f4e8",
            "color_lightblue": "#add8e6",
            "color_lightgrey": "#d3d3d3",
        }
        if data in color_map:
            color_text = color_map[data]
        elif data == "color_custom":
            await safe_edit_or_send(update, _("color", context))
            return States.BG_WAIT_COLOR
        else:
            return States.BG_WAIT_COLOR
    elif update.message:
        color_text = update.message.text.strip()

    if not color_text:
        return States.BG_WAIT_COLOR

    img = context.user_data.get("bg_img")
    if not img:
        await safe_edit_or_send(update, _("error", context))
        release_lock(context)
        await send_main_menu(update, context)
        return ConversationHandler.END

    reply_to = update.message or update.callback_query.message
    msg = await reply_to.reply_text(_("processing", context))

    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, person_segmentation_replace, img, color_text),
            timeout=PROCESSING_TIMEOUT,
        )
        context.user_data["bg_result"] = result
    except asyncio.TimeoutError:
        await msg.edit_text(_("timeout_error", context))
        release_lock(context)
        await send_main_menu(update, context)
        return ConversationHandler.END
    except Exception as e:
        logger.error(f"bg_wait_color uid={uid}: {e}")
        await msg.edit_text(_("error", context))
        release_lock(context)
        await send_main_menu(update, context)
        return ConversationHandler.END

    await msg.delete()
    preview_buf = create_preview(result.copy())
    kb = confirm_kb("bg_ok", "bg_retry", context)
    await reply_to.reply_photo(
        photo=preview_buf,
        caption=_("preview", context) + "\n" + _("bg_warning", context),
        reply_markup=kb,
    )
    cleanup(context, ["bg_img"])
    return States.BG_PREVIEW

async def bg_preview(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "bg_ok":
        try:
            await query.edit_message_caption(caption=_("format_choose", context), reply_markup=format_kb())
        except BadRequest:
            await query.message.reply_text(_("format_choose", context), reply_markup=format_kb())
        return States.BG_WAIT_FORMAT

    elif data == "bg_retry":
        cleanup(context, ["bg_result"])
        try:
            await query.message.delete()
        except Exception:
            pass
        await context.bot.send_message(chat_id=update.effective_chat.id, text=_("send_photo", context))
        release_lock(context)
        if not acquire_lock(context):
            return ConversationHandler.END
        return States.BG_WAIT_PHOTO

    return States.BG_PREVIEW

async def bg_wait_format(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    fmt = query.data.replace("fmt_", "")
    uid = update.effective_user.id

    result = context.user_data.get("bg_result")
    if not result:
        await safe_edit_or_send(update, _("error", context))
        release_lock(context)
        await send_main_menu(update, context)
        return ConversationHandler.END

    try:
        dpi_val = context.user_data.get("dpi", DPI)
        buf = save_image(result, fmt, dpi_val)
        size_str = format_file_size(buf.getbuffer().nbytes)
        caption = (
            f"✅ Background changed! {VERSION}\n"
            f"📏 {result.width}×{result.height} px  |  📦 {size_str}  |  🖼 {fmt}\n"
            f"{_('reminder', context)}"
        )
        context.user_data["last_image"] = buf.getvalue()
        await query.message.reply_document(
            document=io.BytesIO(buf.getvalue()),
            filename=f"bg_changed.{fmt.lower()}",
            caption=caption,
        )
    except Exception as e:
        logger.error(f"bg_wait_format uid={uid}: {e}")
        await safe_edit_or_send(update, _("error", context))
    finally:
        cleanup(context, ["bg_img", "bg_result"])
        release_lock(context)
        await send_main_menu(update, context)

    return ConversationHandler.END

# ─────────────────────────────────────────────────────────────────────
# RESIZE FLOW
# ─────────────────────────────────────────────────────────────────────
async def resize_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "resize_custom":
        if not acquire_lock(context):
            await safe_edit_or_send(update, _("processing", context))
            return ConversationHandler.END
        await safe_edit_or_send(update, _("send_photo", context))
        return States.CUSTOM_WAIT_PHOTO
    elif data == "resize_reduce":
        if not acquire_lock(context):
            await safe_edit_or_send(update, _("processing", context))
            return ConversationHandler.END
        await safe_edit_or_send(update, _("reduce_send_photo", context))
        return States.REDUCE_WAIT_PHOTO

    return States.RESIZE_MODE

# ── Custom Resize ──
async def custom_wait_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    uid = update.effective_user.id
    try:
        img = await get_image_from_update(update)
        if not img:
            await update.message.reply_text(_("send_photo", context))
            return States.CUSTOM_WAIT_PHOTO
        context.user_data["resize_img"] = img
        await update.message.reply_text(_("dimensions", context))
        return States.CUSTOM_WAIT_DIMS
    except Exception as e:
        logger.error(f"custom_wait_photo uid={uid}: {e}")
        await update.message.reply_text(_("error", context))
        release_lock(context)
        await send_main_menu(update, context)
        return ConversationHandler.END

async def custom_wait_dims(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    uid = update.effective_user.id
    dims = parse_dimensions(update.message.text)
    if not dims:
        await update.message.reply_text(_("dimensions", context))
        return States.CUSTOM_WAIT_DIMS

    w, h = dims
    img = context.user_data.get("resize_img")
    if not img:
        await update.message.reply_text(_("error", context))
        release_lock(context)
        await send_main_menu(update, context)
        return ConversationHandler.END

    msg = await update.message.reply_text(_("processing", context))
    try:
        result = img.resize((w, h), Image.Resampling.LANCZOS)
        context.user_data["resize_result"] = result
        await msg.delete()
        preview_buf = create_preview(result.copy())
        kb = confirm_kb("resize_ok", "resize_retry", context)
        await update.message.reply_photo(photo=preview_buf, caption=_("preview", context), reply_markup=kb)
        cleanup(context, ["resize_img"])
        return States.CUSTOM_PREVIEW
    except Exception as e:
        logger.error(f"custom_wait_dims uid={uid}: {e}")
        await msg.edit_text(_("error", context))
        release_lock(context)
        await send_main_menu(update, context)
        return ConversationHandler.END

async def custom_preview(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "resize_ok":
        try:
            await query.edit_message_caption(caption=_("size_option", context), reply_markup=size_option_kb(context))
        except BadRequest:
            await query.message.reply_text(_("size_option", context), reply_markup=size_option_kb(context))
        return States.CUSTOM_WAIT_SIZE_OPT
    elif data == "resize_retry":
        cleanup(context, ["resize_result"])
        try:
            await query.message.delete()
        except Exception:
            pass
        await context.bot.send_message(chat_id=update.effective_chat.id, text=_("send_photo", context))
        release_lock(context)
        if not acquire_lock(context):
            return ConversationHandler.END
        return States.CUSTOM_WAIT_PHOTO

    return States.CUSTOM_PREVIEW

async def custom_size_option(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "size_kb":
        await safe_edit_or_send(update, _("enter_kb", context))
        return States.CUSTOM_WAIT_KB
    elif data == "size_dims":
        await safe_edit_or_send(update, _("format_choose", context), format_kb())
        return States.CUSTOM_WAIT_FORMAT

    return States.CUSTOM_WAIT_SIZE_OPT

async def custom_wait_kb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        target_kb = int(re.sub(r"[^\d]", "", update.message.text))
        if target_kb <= 0:
            raise ValueError
    except (ValueError, TypeError):
        await update.message.reply_text(_("enter_kb", context))
        return States.CUSTOM_WAIT_KB
    context.user_data["target_kb"] = target_kb
    await update.message.reply_text(_("format_choose", context), reply_markup=format_kb())
    return States.CUSTOM_WAIT_FORMAT

async def custom_wait_format(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    fmt = query.data.replace("fmt_", "")
    uid = update.effective_user.id

    result = context.user_data.get("resize_result")
    if not result:
        await safe_edit_or_send(update, _("error", context))
        release_lock(context)
        await send_main_menu(update, context)
        return ConversationHandler.END

    try:
        dpi_val = context.user_data.get("dpi", DPI)
        target_kb = context.user_data.get("target_kb")
        buf = compress_to_kb(result, target_kb, fmt) if target_kb else save_image(result, fmt, dpi_val)
        size_str = format_file_size(buf.getbuffer().nbytes)
        caption = (
            f"✅ Resized! {VERSION}\n"
            f"📏 {result.width}×{result.height} px  |  📦 {size_str}  |  🖼 {fmt}\n"
            f"{_('reminder', context)}"
        )
        context.user_data["last_image"] = buf.getvalue()
        await query.message.reply_document(
            document=io.BytesIO(buf.getvalue()),
            filename=f"resized.{fmt.lower()}",
            caption=caption,
        )
    except Exception as e:
        logger.error(f"custom_wait_format uid={uid}: {e}")
        await safe_edit_or_send(update, _("error", context))
    finally:
        cleanup(context, ["resize_img", "resize_result", "target_kb"])
        release_lock(context)
        await send_main_menu(update, context)

    return ConversationHandler.END

# ── Reduce Size ──
async def reduce_wait_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    uid = update.effective_user.id
    try:
        img = await get_image_from_update(update)
        if not img:
            await update.message.reply_text(_("send_photo", context))
            return States.REDUCE_WAIT_PHOTO
        context.user_data["reduce_img"] = img
        await update.message.reply_text(_("enter_kb", context))
        return States.REDUCE_WAIT_KB
    except Exception as e:
        logger.error(f"reduce_wait_photo uid={uid}: {e}")
        await update.message.reply_text(_("error", context))
        release_lock(context)
        await send_main_menu(update, context)
        return ConversationHandler.END

async def reduce_wait_kb(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        target_kb = int(re.sub(r"[^\d]", "", update.message.text))
        if target_kb <= 0:
            raise ValueError
    except (ValueError, TypeError):
        await update.message.reply_text(_("enter_kb", context))
        return States.REDUCE_WAIT_KB
    context.user_data["target_kb"] = target_kb
    await update.message.reply_text(_("format_choose", context), reply_markup=format_kb())
    return States.REDUCE_WAIT_FORMAT

async def reduce_wait_format(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    fmt = query.data.replace("fmt_", "")
    uid = update.effective_user.id

    img = context.user_data.get("reduce_img")
    target_kb = context.user_data.get("target_kb", 100)
    if not img:
        await safe_edit_or_send(update, _("error", context))
        release_lock(context)
        await send_main_menu(update, context)
        return ConversationHandler.END

    msg = await query.message.reply_text(_("processing", context))
    try:
        buf = compress_to_kb(img, target_kb, fmt)
        size_str = format_file_size(buf.getbuffer().nbytes)
        caption = (
            f"✅ Size reduced! {VERSION}\n"
            f"📏 {img.width}×{img.height} px  |  📦 {size_str}  |  🖼 {fmt}\n"
            f"{_('reminder', context)}"
        )
        context.user_data["last_image"] = buf.getvalue()
        await msg.delete()
        await query.message.reply_document(
            document=io.BytesIO(buf.getvalue()),
            filename=f"reduced.{fmt.lower()}",
            caption=caption,
        )
    except Exception as e:
        logger.error(f"reduce_wait_format uid={uid}: {e}")
        await msg.edit_text(_("error", context))
    finally:
        cleanup(context, ["reduce_img", "target_kb"])
        release_lock(context)
        await send_main_menu(update, context)

    return ConversationHandler.END

# ─────────────────────────────────────────────────────────────────────
# SIGNATURE FLOW
# ─────────────────────────────────────────────────────────────────────
async def sig_wait_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    uid = update.effective_user.id
    try:
        img = await get_image_from_update(update)
        if not img:
            await update.message.reply_text(_("send_photo", context))
            return States.SIG_WAIT_PHOTO

        arr = np.array(img.convert("RGB"))
        corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
        if not all(v > 200 for v in np.mean(corners, axis=0)):
            await update.message.reply_text(_("signature_bg_warn", context))

        msg = await update.message.reply_text(_("processing", context))
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, extract_signature, img)
        context.user_data["sig_result"] = result
        await msg.delete()

        preview_buf = create_preview(result.copy())
        kb = confirm_kb("sig_ok", "sig_retry", context)
        await update.message.reply_photo(photo=preview_buf, caption=_("preview", context), reply_markup=kb)
        return States.SIG_PREVIEW

    except Exception as e:
        logger.error(f"sig_wait_photo uid={uid}: {e}")
        await update.message.reply_text(_("error", context))
        release_lock(context)
        await send_main_menu(update, context)
        return ConversationHandler.END

async def sig_preview(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "sig_ok":
        sig_fmt_kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("PNG", callback_data="fmt_PNG"),
             InlineKeyboardButton("PDF", callback_data="fmt_PDF")],
        ])
        try:
            await query.edit_message_caption(caption=_("format_choose", context), reply_markup=sig_fmt_kb)
        except BadRequest:
            await query.message.reply_text(_("format_choose", context), reply_markup=sig_fmt_kb)
        return States.SIG_WAIT_FORMAT
    elif data == "sig_retry":
        cleanup(context, ["sig_result"])
        try:
            await query.message.delete()
        except Exception:
            pass
        await context.bot.send_message(chat_id=update.effective_chat.id, text=_("send_photo", context))
        return States.SIG_WAIT_PHOTO

    return States.SIG_PREVIEW

async def sig_wait_format(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    fmt = query.data.replace("fmt_", "")
    uid = update.effective_user.id

    result = context.user_data.get("sig_result")
    if not result:
        await safe_edit_or_send(update, _("error", context))
        release_lock(context)
        await send_main_menu(update, context)
        return ConversationHandler.END

    try:
        buf = io.BytesIO()
        if fmt == "PNG":
            result.save(buf, format="PNG")
        else:
            result.convert("RGB").save(buf, format="PDF")
        buf.seek(0)
        size_str = format_file_size(buf.getbuffer().nbytes)
        caption = (
            f"✅ Signature extracted! {VERSION}\n"
            f"📦 {size_str}  |  🖼 {fmt}\n"
            f"{_('reminder', context)}"
        )
        context.user_data["last_image"] = buf.getvalue()
        await query.message.reply_document(
            document=io.BytesIO(buf.getvalue()),
            filename=f"signature.{fmt.lower()}",
            caption=caption,
        )
    except Exception as e:
        logger.error(f"sig_wait_format uid={uid}: {e}")
        await safe_edit_or_send(update, _("error", context))
    finally:
        cleanup(context, ["sig_result"])
        release_lock(context)
        await send_main_menu(update, context)

    return ConversationHandler.END

# ─────────────────────────────────────────────────────────────────────
# FALLBACKS & ERROR HANDLER
# ─────────────────────────────────────────────────────────────────────
async def conversation_fallback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message:
        await update.message.reply_text(_("unexpected", context))
    return ConversationHandler.END

async def global_fallback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_main_menu(update, context)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update error: {context.error}", exc_info=context.error)
    if isinstance(context.error, (TimedOut, NetworkError)):
        return
    if update and hasattr(update, "effective_chat") and update.effective_chat:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⚠️ An unexpected error occurred. Please try /start again.",
            )
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────
# GRACEFUL SHUTDOWN
# ─────────────────────────────────────────────────────────────────────
def handle_signal(sig, frame):
    logger.info(f"Signal {sig} received. Shutting down...")
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    token = os.environ.get("BOT_TOKEN")
    if not token:
        logger.error("BOT_TOKEN not set. Exiting.")
        sys.exit(1)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info(f"Flask health server started on port {os.environ.get('PORT', 8080)}")

    application = Application.builder().token(token).build()

    # Standalone commands (work outside conversation too)
    application.add_handler(CommandHandler("hinglish", cmd_hinglish))
    application.add_handler(CommandHandler("strict", cmd_strict))
    application.add_handler(CommandHandler("dpi", cmd_dpi))
    application.add_handler(CommandHandler("history", cmd_history))

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            States.SELECT_ACTION:        [CallbackQueryHandler(select_action)],
            States.BG_WAIT_PHOTO:        [MessageHandler(filters.PHOTO | filters.Document.IMAGE, bg_wait_photo)],
            States.BG_WAIT_COLOR:        [
                CallbackQueryHandler(bg_wait_color, pattern="^color_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, bg_wait_color),
            ],
            States.BG_PREVIEW:           [CallbackQueryHandler(bg_preview, pattern="^bg_")],
            States.BG_WAIT_FORMAT:       [CallbackQueryHandler(bg_wait_format, pattern="^fmt_")],
            States.RESIZE_MODE:          [CallbackQueryHandler(resize_mode, pattern="^resize_")],
            States.CUSTOM_WAIT_PHOTO:    [MessageHandler(filters.PHOTO | filters.Document.IMAGE, custom_wait_photo)],
            States.CUSTOM_WAIT_DIMS:     [MessageHandler(filters.TEXT & ~filters.COMMAND, custom_wait_dims)],
            States.CUSTOM_PREVIEW:       [CallbackQueryHandler(custom_preview, pattern="^resize_")],
            States.CUSTOM_WAIT_SIZE_OPT: [CallbackQueryHandler(custom_size_option, pattern="^size_")],
            States.CUSTOM_WAIT_KB:       [MessageHandler(filters.TEXT & ~filters.COMMAND, custom_wait_kb)],
            States.CUSTOM_WAIT_FORMAT:   [CallbackQueryHandler(custom_wait_format, pattern="^fmt_")],
            States.REDUCE_WAIT_PHOTO:    [MessageHandler(filters.PHOTO | filters.Document.IMAGE, reduce_wait_photo)],
            States.REDUCE_WAIT_KB:       [MessageHandler(filters.TEXT & ~filters.COMMAND, reduce_wait_kb)],
            States.REDUCE_WAIT_FORMAT:   [CallbackQueryHandler(reduce_wait_format, pattern="^fmt_")],
            States.SIG_WAIT_PHOTO:       [MessageHandler(filters.PHOTO | filters.Document.IMAGE, sig_wait_photo)],
            States.SIG_PREVIEW:          [CallbackQueryHandler(sig_preview, pattern="^sig_")],
            States.SIG_WAIT_FORMAT:      [CallbackQueryHandler(sig_wait_format, pattern="^fmt_")],
        },
        fallbacks=[
            CommandHandler("cancel", cmd_cancel),
            CommandHandler("start", start),
            MessageHandler(filters.ALL, conversation_fallback),
        ],
        allow_reentry=True,
    )

    application.add_handler(conv)
    application.add_handler(MessageHandler(filters.ALL, global_fallback))
    application.add_error_handler(error_handler)

    logger.info(f"Bot {VERSION} starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
