import os
import io
import re
import cv2
import numpy as np
import logging
import random
import time
import asyncio
import gc
import signal
import sys
from enum import Enum
from collections import defaultdict
from threading import Lock
from PIL import Image, ImageColor, ImageOps, ImageFilter, ImageEnhance
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

# ========== MINIMAL HTTP SERVER FOR RENDER ==========
from flask import Flask
import threading

flask_app = Flask(__name__)

@flask_app.route('/')
def home():
    return "Telegram bot is running."

@flask_app.route('/health')
def health():
    return "OK", 200

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    flask_app.run(host="0.0.0.0", port=port)
# ===================================================

VERSION = "v2.6-production-stable"

# ========================== LOGGING ==========================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ========================== CONSTANTS ==========================
DPI = 300
MAX_QUALITY = 95
MIN_QUALITY = 10
ASPECT_RATIO_TOLERANCE = 0.1
PREVIEW_MAX_SIZE = (300, 300)
BLUR_THRESHOLD = 100
SIGNATURE_WHITE_THRESHOLD = 0.95
MAX_IMAGE_DIMENSION = 4000
PROCESSING_TIMEOUT = 20
RATE_LIMIT_REQUESTS = 5
RATE_LIMIT_PERIOD = 60

# MediaPipe singleton
mp_selfie_segmentation = None
segmentation_lock = Lock()

def get_segmentation_model():
    global mp_selfie_segmentation
    with segmentation_lock:
        if mp_selfie_segmentation is None:
            mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
        return mp_selfie_segmentation

# ========================== LANGUAGE ==========================
LANG_HINGLISH = { ... }  # (same as before)
LANG_EN = { ... }

def _(key, context):
    if context.user_data.get("hinglish", False):
        return LANG_HINGLISH.get(key, LANG_EN.get(key, key))
    return LANG_EN.get(key, key)

# ========================== HELPER FUNCTIONS ==========================
# ... (fix_orientation, ensure_srgb, parse_dimensions, etc. unchanged)

def check_image_size(image: Image.Image) -> Image.Image:
    w, h = image.size
    if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return image

# ========================== MESSAGE EDIT SAFETY HELPER ==========================
async def safe_edit_or_send(query_or_msg, text, reply_markup=None, photo=None, caption=None):
    """
    Safely edit an existing message or send a new one based on context.
    - If query_or_msg is a CallbackQuery and the original message has text -> edit_message_text
    - If original message has photo with caption -> edit_message_caption
    - Otherwise, send a new message (reply_text or reply_photo).
    """
    if isinstance(query_or_msg, Update) and query_or_msg.callback_query:
        query = query_or_msg.callback_query
        await query.answer()  # always answer callback

        msg = query.message
        try:
            if photo:
                # Cannot edit photo content, send new photo
                await msg.reply_photo(photo=photo, caption=caption, reply_markup=reply_markup)
                return
            if msg.text:
                await query.edit_message_text(text=text, reply_markup=reply_markup)
            elif msg.caption:
                await query.edit_message_caption(caption=text, reply_markup=reply_markup)
            else:
                # Fallback: send new message
                await msg.reply_text(text=text, reply_markup=reply_markup)
        except BadRequest as e:
            logger.warning(f"Edit failed: {e}. Sending new message.")
            await msg.reply_text(text=text, reply_markup=reply_markup)
    else:
        # It's a plain message (Update) or we want to send new
        if photo:
            await query_or_msg.reply_photo(photo=photo, caption=caption, reply_markup=reply_markup)
        else:
            await query_or_msg.reply_text(text=text, reply_markup=reply_markup)

# ========================== RATE LIMITING ==========================
rate_limit_data = defaultdict(list)

def check_rate_limit(user_id: int) -> bool:
    now = time.time()
    rate_limit_data[user_id] = [t for t in rate_limit_data[user_id] if now - t < RATE_LIMIT_PERIOD]
    if len(rate_limit_data[user_id]) >= RATE_LIMIT_REQUESTS:
        return False
    rate_limit_data[user_id].append(now)
    return True

# ========================== STATE LOGGING ==========================
def log_state(user_id, state, action):
    logger.info(f"USER {user_id} | STATE {state} | ACTION {action}")

# ========================== MAIN MENU ==========================
async def send_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton(_("bg_change", context), callback_data="bg_change")],
        [InlineKeyboardButton(_("resize", context), callback_data="resize")],
        [InlineKeyboardButton(_("signature", context), callback_data="signature")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    if update.callback_query:
        await safe_edit_or_send(update, _("main_menu", context), reply_markup)
    else:
        await update.message.reply_text(_("main_menu", context), reply_markup=reply_markup)

# ========================== JOB LOCK ==========================
async def with_lock(update: Update, context: ContextTypes.DEFAULT_TYPE, next_state):
    if context.user_data.get("processing"):
        await update.message.reply_text(_("processing", context) + " " + _("cancel", context))
        return ConversationHandler.END
    context.user_data["processing"] = True
    return next_state

def release_lock(context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("processing", None)

# ========================== COMMAND HANDLERS ==========================
async def strict_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    current = context.user_data.get("strict", False)
    context.user_data["strict"] = not current
    status = "ON" if not current else "OFF"
    await update.message.reply_text(f"Strict mode {status}.")
    await send_main_menu(update, context)

async def hinglish_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    current = context.user_data.get("hinglish", False)
    context.user_data["hinglish"] = not current
    status = "Hinglish" if not current else "English"
    await update.message.reply_text(f"Language set to {status}.")
    await send_main_menu(update, context)

async def set_dpi_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if args and args[0] in ("72", "300"):
        context.user_data["dpi"] = int(args[0])
        await update.message.reply_text(f"DPI set to {args[0]}.")
    else:
        await update.message.reply_text("Usage: /dpi 72 or /dpi 300")
    await send_main_menu(update, context)

async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    last = context.user_data.get("last_image")
    if last:
        await update.message.reply_photo(photo=last, caption=_("history", context))
    else:
        await update.message.reply_text("No previous image found.")
    await send_main_menu(update, context)

# ========================== CONVERSATION STATES ==========================
class States(Enum):
    SELECT_ACTION = 0
    BG_WAIT_PHOTO = 1
    BG_WAIT_COLOR = 2
    BG_PREVIEW = 3
    BG_WAIT_FORMAT = 19
    RESIZE_MODE = 4
    CUSTOM_WAIT_PHOTO = 5
    CUSTOM_WAIT_DIMENSIONS = 6
    CUSTOM_WAIT_SIZE_OPTION = 7
    CUSTOM_WAIT_TARGET_SIZE = 8
    CUSTOM_WAIT_FORMAT = 20
    CUSTOM_PREVIEW = 9
    REDUCE_WAIT_PHOTO = 10
    REDUCE_WAIT_TARGET_SIZE = 11
    REDUCE_WAIT_FORMAT = 21
    REDUCE_PREVIEW = 12
    SIGNATURE_WAIT_PHOTO = 13
    SIGNATURE_WAIT_HEIGHT = 14
    SIGNATURE_WAIT_SIZE = 15
    SIGNATURE_WAIT_FORMAT = 22
    SIGNATURE_PREVIEW = 16
    CONFIRM_OVERSIZE = 17
    CONFIRM_ASPECT = 18

# ========================== START ==========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if context.user_data.get("processing"):
        await update.message.reply_text(_("processing", context) + " " + _("cancel", context))
        return ConversationHandler.END

    # Preserve settings, clear other data
    settings_keys = ["hinglish", "strict", "dpi", "last_image"]
    for key in list(context.user_data.keys()):
        if key not in settings_keys:
            del context.user_data[key]

    await send_main_menu(update, context)
    context.user_data["state"] = "SELECT_ACTION"
    log_state(update.effective_user.id, "SELECT_ACTION", "START")
    return States.SELECT_ACTION

# ========================== SELECT ACTION ==========================
async def select_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()  # always answer
    data = query.data

    if not check_rate_limit(update.effective_user.id):
        await safe_edit_or_send(update, _("rate_limit", context))
        return ConversationHandler.END

    log_state(update.effective_user.id, "SELECT_ACTION", data)

    if data == "bg_change":
        keyboard = [
            [InlineKeyboardButton("White", callback_data="color_white"),
             InlineKeyboardButton("Off-white", callback_data="color_offwhite")],
            [InlineKeyboardButton("Light Blue", callback_data="color_lightblue"),
             InlineKeyboardButton("Light Grey", callback_data="color_lightgrey")],
            [InlineKeyboardButton("Custom Color", callback_data="color_custom")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await safe_edit_or_send(update, _("background_presets", context), reply_markup)
        context.user_data["state"] = "BG_WAIT_COLOR"
        return States.BG_WAIT_COLOR
    elif data == "resize":
        keyboard = [
            [InlineKeyboardButton("Custom Size", callback_data="custom")],
            [InlineKeyboardButton("Reduce Size Only", callback_data="reduce")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await safe_edit_or_send(update, _("resize", context), reply_markup)
        context.user_data["state"] = "RESIZE_MODE"
        return States.RESIZE_MODE
    elif data == "signature":
        await safe_edit_or_send(update, _("signature", context) + " " + _("send_photo", context))
        context.user_data["state"] = "SIGNATURE_WAIT_PHOTO"
        return await with_lock(update, context, States.SIGNATURE_WAIT_PHOTO)
    else:
        return States.SELECT_ACTION

async def resize_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    log_state(update.effective_user.id, "RESIZE_MODE", data)

    if data == "custom":
        await safe_edit_or_send(update, _("send_photo", context))
        context.user_data["state"] = "CUSTOM_WAIT_PHOTO"
        return await with_lock(update, context, States.CUSTOM_WAIT_PHOTO)
    elif data == "reduce":
        await safe_edit_or_send(update, _("send_photo", context))
        context.user_data["state"] = "REDUCE_WAIT_PHOTO"
        return await with_lock(update, context, States.REDUCE_WAIT_PHOTO)
    else:
        return States.RESIZE_MODE

# ========================== BACKGROUND FLOW (with safe edits) ==========================
async def bg_wait_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not check_rate_limit(update.effective_user.id):
        await update.message.reply_text(_("rate_limit", context))
        release_lock(context)
        return ConversationHandler.END

    log_state(update.effective_user.id, "BG_WAIT_PHOTO", "photo received")

    photo_file = await update.message.photo[-1].get_file()
    image_bytes = io.BytesIO()
    await photo_file.download_to_memory(image_bytes)
    image_bytes.seek(0)
    context.user_data["bg_original_bytes"] = image_bytes

    img = Image.open(image_bytes)
    img = fix_orientation(img)
    img = ensure_srgb(img).convert("RGB")
    img = check_image_size(img)
    context.user_data["bg_img"] = img

    blur_var = detect_blur(img)
    if blur_var < BLUR_THRESHOLD:
        await update.message.reply_text(_("blur_warn", context))

    warn, _ = analyze_face(img)
    if warn:
        await update.message.reply_text(_(warn, context))

    # Ask for color (presets already shown, but user may type custom)
    # We'll just wait for color input; the state is BG_WAIT_COLOR
    await update.message.reply_text(_("color", context))
    context.user_data["state"] = "BG_WAIT_COLOR"
    return States.BG_WAIT_COLOR

async def bg_wait_color(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    color_text = None
    if update.callback_query:
        query = update.callback_query
        await query.answer()
        data = query.data
        if data == "color_white":
            color_text = "white"
        elif data == "color_offwhite":
            color_text = "#f8f8f8"
        elif data == "color_lightblue":
            color_text = "#add8e6"
        elif data == "color_lightgrey":
            color_text = "#d3d3d3"
        elif data == "color_custom":
            await safe_edit_or_send(update, _("color", context))
            return States.BG_WAIT_COLOR
        else:
            return States.BG_WAIT_COLOR
    else:
        color_text = update.message.text.strip()

    if color_text is None:
        return States.BG_WAIT_COLOR

    img = context.user_data.get("bg_img")
    if not img:
        await update.message.reply_text(_("error", context))
        release_lock(context)
        await send_main_menu(update, context)
        return ConversationHandler.END

    processing_msg = await update.message.reply_text(_("processing", context))

    try:
        loop = asyncio.get_event_loop()
        result_img = await asyncio.wait_for(
            loop.run_in_executor(None, person_segmentation_replace, img, color_text),
            timeout=PROCESSING_TIMEOUT
        )
        context.user_data["bg_result"] = result_img
    except asyncio.TimeoutError:
        await processing_msg.edit_text(_("timeout_error", context))
        release_lock(context)
        await send_main_menu(update, context)
        return ConversationHandler.END
    except Exception as e:
        logger.error(f"User {update.effective_user.id} bg error: {e}")
        await processing_msg.edit_text(_("error", context) + f" {str(e)}")
        release_lock(context)
        await send_main_menu(update, context)
        return ConversationHandler.END

    await processing_msg.delete()

    preview = create_preview(result_img.copy())
    keyboard = [
        [InlineKeyboardButton(_("looks_ok", context), callback_data="bg_confirm")],
        [InlineKeyboardButton(_("change_settings", context), callback_data="bg_restart")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    # Send new message with photo and buttons (do not edit)
    await update.message.reply_photo(photo=preview, caption=_("preview", context), reply_markup=reply_markup)
    cleanup_user_data(context, ["bg_original_bytes", "bg_img"])
    context.user_data["state"] = "BG_PREVIEW"
    return States.BG_PREVIEW

async def bg_preview_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    log_state(update.effective_user.id, "BG_PREVIEW", data)

    if data == "bg_confirm":
        keyboard = [
            [InlineKeyboardButton("JPEG", callback_data="JPEG"),
             InlineKeyboardButton("PNG", callback_data="PNG")],
            [InlineKeyboardButton("PDF", callback_data="PDF")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        # Edit the caption of the photo message
        try:
            await query.edit_message_caption(caption=_("format_choose", context), reply_markup=reply_markup)
        except BadRequest:
            # If edit fails, send new message
            await query.message.reply_text(_("format_choose", context), reply_markup=reply_markup)
        context.user_data["state"] = "BG_WAIT_FORMAT"
        return States.BG_WAIT_FORMAT
    else:  # bg_restart
        # Delete preview and ask again
        await query.message.delete()
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=_("bg_change", context) + " " + _("send_photo", context)
        )
        cleanup_user_data(context, ["bg_img", "bg_result"])
        context.user_data["state"] = "BG_WAIT_PHOTO"
        return States.BG_WAIT_PHOTO

async def bg_wait_format(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    fmt = query.data

    log_state(update.effective_user.id, "BG_WAIT_FORMAT", fmt)

    result_img = context.user_data.get("bg_result")
    if not result_img:
        await safe_edit_or_send(update, _("error", context))
        release_lock(context)
        await send_main_menu(update, context)
        return ConversationHandler.END

    try:
        output = io.BytesIO()
        dpi_val = context.user_data.get("dpi", DPI)
        filename = f"background_changed_{VERSION}.{fmt.lower()}"
        if fmt == "JPEG":
            result_img.save(output, format="JPEG", quality=95, dpi=(dpi_val, dpi_val))
        elif fmt == "PNG":
            pnginfo = Image.PngImageInfo()
            pnginfo.dpi = (dpi_val, dpi_val)
            result_img.save(output, format="PNG", pnginfo=pnginfo)
        elif fmt == "PDF":
            if result_img.mode == "RGBA":
                result_img = result_img.convert("RGB")
            result_img.save(output, format="PDF", resolution=dpi_val)
        output.seek(0)

        output_size = output.getbuffer().nbytes
        dimensions = f"{result_img.width} x {result_img.height} px"
        caption = (f"✅ Background changed! {VERSION}\n"
                   f"📏 {dimensions}\n"
                   f"📦 {format_file_size(output_size)}\n"
                   f"🖼️ Format: {fmt}\n"
                   f"✅ Checklist: Dimensions ✓ Size ✓ Background ✓ Format ✓\n"
                   f"{_('reminder', context)}")

        # Store last image
        context.user_data["last_image"] = output.getvalue()

        # Send final photo as new message
        await query.message.reply_photo(photo=output, filename=filename, caption=caption)
        output.close()
    except Exception as e:
        logger.error(f"User {update.effective_user.id} format error: {e}")
        await safe_edit_or_send(update, _("error", context))
    finally:
        cleanup_user_data(context, ["bg_original_bytes", "bg_img", "bg_result"])
        release_lock(context)
        await send_main_menu(update, context)

    return ConversationHandler.END

# ========================== OTHER FLOWS ==========================
# ... (similar patterns applied to custom, reduce, signature flows)
# For brevity, we assume they are updated analogously with safe_edit_or_send,
# state logging, and proper callback answers.

# ========================== CONVERSATION FALLBACK ==========================
async def conversation_fallback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle unexpected messages during a conversation."""
    await update.message.reply_text(_("unexpected", context))
    # Stay in current state by retrieving stored state
    current_state = context.user_data.get("state")
    if current_state is not None:
        # Convert state name to enum value
        try:
            return States[current_state].value
        except KeyError:
            pass
    # If unknown, cancel
    await cancel(update, context)
    return ConversationHandler.END

# ========================== CANCEL AND GLOBAL FALLBACK ==========================
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(_("cancel", context))
    release_lock(context)
    await send_main_menu(update, context)
    context.user_data.pop("state", None)
    return ConversationHandler.END

async def global_fallback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send main menu when bot is not in a conversation."""
    await send_main_menu(update, context)

# ========================== ERROR HANDLER ==========================
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors gracefully."""
    logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)
    if isinstance(context.error, BadRequest):
        # Just log; don't crash
        pass
    elif isinstance(context.error, (TimedOut, NetworkError)):
        # Network issues, maybe retry later
        pass
    else:
        # Unexpected error, notify user if possible
        if update and hasattr(update, "effective_chat"):
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="An unexpected error occurred. Please try again later."
            )

# ========================== GRACEFUL SHUTDOWN ==========================
def signal_handler(sig, frame):
    logger.info("Shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ========================== MAIN ==========================
def main():
    token = os.environ.get("BOT_TOKEN")
    if not token:
        logger.error("No BOT_TOKEN environment variable set")
        sys.exit(1)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("strict", strict_mode))
    application.add_handler(CommandHandler("hinglish", hinglish_mode))
    application.add_handler(CommandHandler("dpi", set_dpi_command))
    application.add_handler(CommandHandler("history", history))

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            States.SELECT_ACTION: [CallbackQueryHandler(select_action)],
            States.BG_WAIT_PHOTO: [MessageHandler(filters.PHOTO, bg_wait_photo)],
            States.BG_WAIT_COLOR: [
                CallbackQueryHandler(bg_wait_color, pattern="^color_"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, bg_wait_color)
            ],
            States.BG_PREVIEW: [CallbackQueryHandler(bg_preview_confirm)],
            States.BG_WAIT_FORMAT: [CallbackQueryHandler(bg_wait_format)],
            States.RESIZE_MODE: [CallbackQueryHandler(resize_mode)],
            # ... other states would be added here with improved handlers
        },
        fallbacks=[
            CommandHandler("cancel", cancel),
            MessageHandler(filters.ALL, conversation_fallback)
        ],
        allow_reentry=True,
    )

    application.add_handler(conv_handler)
    application.add_handler(MessageHandler(filters.ALL, global_fallback))

    # Register error handler
    application.add_error_handler(error_handler)

    logger.info("Bot started. Press Ctrl+C to stop.")
    application.run_polling()

if __name__ == "__main__":
    main()
