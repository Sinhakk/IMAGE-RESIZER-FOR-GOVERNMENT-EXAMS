import os
import io
import re
import cv2
import numpy as np
import logging
from enum import Enum
from PIL import Image, ImageColor, ImageOps, ImageFilter
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

# ========== FLASK ADDITION ==========
from flask import Flask
import threading

flask_app = Flask(__name__)

@flask_app.route('/')
def home():
    return "Telegram bot is running."

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    flask_app.run(host="0.0.0.0", port=port)
# ====================================

# ========================== VERSION ==========================
VERSION = "v2.1"

# ========================== LOGGING ==========================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ========================== CONSTANTS ==========================
DPI = 300
MAX_QUALITY = 95
MIN_QUALITY = 10
TARGET_SIZE_TOLERANCE = 0.99
ASPECT_RATIO_TOLERANCE = 0.1
PREVIEW_MAX_SIZE = (300, 300)
BLUR_THRESHOLD = 100
SIGNATURE_WHITE_THRESHOLD = 0.95

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ========================== LANGUAGE SUPPORT ==========================
LANG_HINGLISH = {
    "start": "Namaste! Kya karna chahenge?",
    "bg_change": "Background badalna hai? Photo bhejo.",
    "resize": "Resize karna hai?",
    "signature": "Signature banana hai?",
    "send_photo": "Photo bhejo",
    "processing": "Thoda intezaar karein...",
    "preview": "Preview dekh lo. Theek hai?",
    "looks_ok": "✅ Theek hai",
    "change_settings": "🔄 Badalna hai",
    "format_choose": "Format chuno:",
    "size_target": "File size kitna chahiye? (KB mein)",
    "dimensions": "Dimensions do (jaise 300x200 px, 5x4 cm)",
    "color": "Rang batao (naam ya hex code)",
    "error": "Kuch gadbad hui. Phir se try karo.",
    "cancel": "Cancel kar diya.",
    "blur_warn": "⚠️ Photo thodi blur lag rahi hai. Better photo use karo to avoid rejection.",
    "signature_bg_warn": "⚠️ Signature background safed nahi hai. Safed paper use karna better rahega.",
    "reminder": "⚠️ Form upload se pehle ek baar preview check kar lena.",
}

LANG_EN = {
    "start": "Welcome! What would you like to do?",
    "bg_change": "Change background? Send photo.",
    "resize": "Resize image?",
    "signature": "Create signature?",
    "send_photo": "Send photo",
    "processing": "Processing... please wait.",
    "preview": "Preview the result. Looks OK?",
    "looks_ok": "✅ Looks OK",
    "change_settings": "🔄 Change Settings",
    "format_choose": "Choose format:",
    "size_target": "Enter desired file size in KB:",
    "dimensions": "Provide dimensions (e.g., 300x200 px, 5x4 cm)",
    "color": "Tell me the color (name or hex):",
    "error": "Something went wrong. Please try again.",
    "cancel": "Cancelled.",
    "blur_warn": "⚠️ The photo appears slightly blurry. For best results, use a clearer image.",
    "signature_bg_warn": "⚠️ Signature background is not pure white. Plain white paper is recommended.",
    "reminder": "⚠️ Please double-check the output before uploading to the form.",
}

def _(key, context):
    if context.user_data.get("hinglish", False):
        return LANG_HINGLISH.get(key, LANG_EN.get(key, key))
    return LANG_EN.get(key, key)

# ========================== HELPER FUNCTIONS ==========================
def fix_orientation(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image)

def ensure_srgb(image: Image.Image) -> Image.Image:
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA" if image.mode == "P" and image.info.get("transparency") else "RGB")
    image.info.pop("icc_profile", None)
    return image

def set_dpi(image: Image.Image, dpi=DPI):
    image.info["dpi"] = (dpi, dpi)
    return image

def parse_dimensions(text: str):
    pattern = r"(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*(px|cm|mm|inch|in)?"
    match = re.search(pattern, text)
    if not match:
        raise ValueError("Invalid format. Use like: 300x200 px, 5x4 cm, 2x3 inch")
    w = float(match.group(1))
    h = float(match.group(2))
    unit = match.group(3) or "px"
    return w, h, unit.lower()

def dimensions_to_pixels(width, height, unit):
    if unit == "px":
        return int(width), int(height)
    elif unit in ("cm", "mm", "inch", "in"):
        if unit == "cm":
            inches_w = width / 2.54
            inches_h = height / 2.54
        elif unit == "mm":
            inches_w = width / 25.4
            inches_h = height / 25.4
        else:
            inches_w = width
            inches_h = height
        return int(inches_w * DPI), int(inches_h * DPI)
    else:
        raise ValueError(f"Unknown unit: {unit}")

def resize_to_exact(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    return image.resize((target_w, target_h), Image.Resampling.LANCZOS)

def compress_to_target_size(image: Image.Image, target_kb: int, format: str = "JPEG", strict: bool = False) -> io.BytesIO:
    target_bytes = target_kb * 1024
    tolerance = 0.01 if strict else 0.05
    output = io.BytesIO()

    if format.upper() == "JPEG":
        low, high = MIN_QUALITY, MAX_QUALITY
        best_quality = high
        best_data = None
        while low <= high:
            mid = (low + high) // 2
            img_byte_arr = io.BytesIO()
            if image.mode in ("RGBA", "P", "LA"):
                image = image.convert("RGB")
            image.save(img_byte_arr, format='JPEG', quality=mid, optimize=True, dpi=(DPI, DPI))
            size = img_byte_arr.tell()
            if size <= target_bytes * (1 + tolerance):
                best_quality = mid
                best_data = img_byte_arr.getvalue()
                low = mid + 1
            else:
                high = mid - 1
        if best_data is None:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=MIN_QUALITY, optimize=True, dpi=(DPI, DPI))
            best_data = img_byte_arr.getvalue()
        output = io.BytesIO(best_data)
    elif format.upper() == "PNG":
        pnginfo = Image.PngImageInfo()
        pnginfo.dpi = (DPI, DPI)
        image.save(output, format='PNG', optimize=True, pnginfo=pnginfo)
    elif format.upper() == "PDF":
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(output, format='PDF', resolution=DPI)
    else:
        raise ValueError(f"Unsupported format: {format}")
    output.seek(0)
    return output

def detect_face_center(image: Image.Image):
    open_cv_image = np.array(image.convert('RGB'))
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    center_x = x + w // 2
    center_y = y + h // 2
    return center_x, center_y, w, h

def center_face_in_dimensions(image: Image.Image, target_w: int, target_h: int):
    orig_w, orig_h = image.size
    face_data = detect_face_center(image)
    if face_data is None:
        target_ratio = target_w / target_h
        orig_ratio = orig_w / orig_h
        if orig_ratio > target_ratio:
            new_w = int(orig_h * target_ratio)
            left = (orig_w - new_w) // 2
            image = image.crop((left, 0, left + new_w, orig_h))
        else:
            new_h = int(orig_w / target_ratio)
            top = (orig_h - new_h) // 2
            image = image.crop((0, top, orig_w, top + new_h))
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    cx, cy, fw, fh = face_data
    target_face_y = int(target_h * 0.45)
    target_face_height = int(target_h * 0.6)
    scale = target_face_height / fh
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    new_cx = int(cx * scale)
    new_cy = int(cy * scale)
    left = new_cx - target_w // 2
    top = new_cy - target_face_y
    left = max(0, min(left, new_w - target_w))
    top = max(0, min(top, new_h - target_h))
    cropped = resized.crop((left, top, left + target_w, top + target_h))
    return cropped

def simple_background_replace(image: Image.Image, new_color):
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_array = np.array(image)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lower = np.array([0, 0, 200])
    upper = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    mask = mask / 255.0

    if isinstance(new_color, str):
        try:
            new_color_rgb = ImageColor.getrgb(new_color)
        except ValueError:
            raise ValueError("Invalid color name or hex code")
    else:
        new_color_rgb = new_color
    new_bg = np.full_like(img_array, new_color_rgb)

    mask_3ch = np.stack([mask, mask, mask], axis=2)
    result = (img_array * (1 - mask_3ch) + new_bg * mask_3ch).astype(np.uint8)
    return Image.fromarray(result)

def process_signature(image: Image.Image, target_height: int, target_kb: int, strict: bool = False) -> io.BytesIO:
    if image.mode != "L":
        image = image.convert("L")
    image = ImageOps.autocontrast(image, cutoff=2)
    threshold = 128
    image = image.point(lambda p: 255 if p > threshold else 0)
    image = image.convert("RGB")
    w, h = image.size
    new_w = int(w * target_height / h)
    image = image.resize((new_w, target_height), Image.Resampling.LANCZOS)
    return compress_to_target_size(image, target_kb, "JPEG", strict)

def format_file_size(size_bytes):
    return f"{size_bytes/1024:.1f} KB"

def create_preview(image: Image.Image) -> io.BytesIO:
    image.thumbnail(PREVIEW_MAX_SIZE, Image.Resampling.LANCZOS)
    preview = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(preview, format="JPEG", quality=70)
    preview.seek(0)
    return preview

def check_aspect_ratio(orig_w, orig_h, target_w, target_h):
    orig_ratio = orig_w / orig_h
    target_ratio = target_w / target_h
    diff = abs(orig_ratio - target_ratio) / target_ratio
    return diff <= ASPECT_RATIO_TOLERANCE

def detect_blur(image: Image.Image) -> float:
    gray = np.array(image.convert("L"))
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

def auto_level(image: Image.Image) -> Image.Image:
    return ImageOps.autocontrast(image, cutoff=1)

def check_signature_bg(image: Image.Image) -> bool:
    img_array = np.array(image.convert("RGB"))
    white_mask = np.all(img_array > 240, axis=2)
    white_ratio = np.sum(white_mask) / white_mask.size
    return white_ratio >= SIGNATURE_WHITE_THRESHOLD

# ========================== JOB LOCK ==========================
async def with_lock(update: Update, context: ContextTypes.DEFAULT_TYPE, next_state):
    if context.user_data.get("processing"):
        await update.message.reply_text(_("processing", context) + " " + _("cancel", context))
        return ConversationHandler.END
    context.user_data["processing"] = True
    return next_state

def release_lock(context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("processing", None)

# ========================== COMMAND HANDLERS (SETTINGS) ==========================
async def strict_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    current = context.user_data.get("strict", False)
    context.user_data["strict"] = not current
    status = "ON" if not current else "OFF"
    await update.message.reply_text(f"Strict mode {status}. File size tolerance now ±1 KB when ON.")

async def hinglish_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    current = context.user_data.get("hinglish", False)
    context.user_data["hinglish"] = not current
    status = "Hinglish" if not current else "English"
    await update.message.reply_text(f"Language set to {status}.")

async def set_dpi_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if args and args[0] in ("72", "300"):
        context.user_data["dpi"] = int(args[0])
        await update.message.reply_text(f"DPI set to {args[0]}.")
    else:
        await update.message.reply_text("Usage: /dpi 72 or /dpi 300")

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

# ========================== START AND ACTION ==========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Welcome message with action selection."""
    if context.user_data.get("processing"):
        await update.message.reply_text(_("processing", context) + " " + _("cancel", context))
        return ConversationHandler.END

    keyboard = [
        [InlineKeyboardButton(_("bg_change", context), callback_data="bg_change")],
        [InlineKeyboardButton(_("resize", context), callback_data="resize")],
        [InlineKeyboardButton(_("signature", context), callback_data="signature")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        _("start", context),
        reply_markup=reply_markup,
    )
    return States.SELECT_ACTION

async def select_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "bg_change":
        await query.edit_message_text(_("bg_change", context) + " " + _("send_photo", context))
        return await with_lock(update, context, States.BG_WAIT_PHOTO)
    elif data == "resize":
        keyboard = [
            [InlineKeyboardButton("Custom Size", callback_data="custom")],
            [InlineKeyboardButton("Reduce Size Only", callback_data="reduce")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(_("resize", context), reply_markup=reply_markup)
        return States.RESIZE_MODE
    elif data == "signature":
        await query.edit_message_text(_("signature", context) + " " + _("send_photo", context))
        return await with_lock(update, context, States.SIGNATURE_WAIT_PHOTO)
    else:
        return States.SELECT_ACTION

async def resize_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "custom":
        await query.edit_message_text(_("send_photo", context))
        return await with_lock(update, context, States.CUSTOM_WAIT_PHOTO)
    elif data == "reduce":
        await query.edit_message_text(_("send_photo", context))
        return await with_lock(update, context, States.REDUCE_WAIT_PHOTO)
    else:
        return States.RESIZE_MODE

# ========================== BACKGROUND CHANGE FLOW ==========================
async def bg_wait_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    photo_file = await update.message.photo[-1].get_file()
    image_bytes = io.BytesIO()
    await photo_file.download_to_memory(image_bytes)
    image_bytes.seek(0)
    context.user_data["bg_original"] = image_bytes

    img = Image.open(image_bytes)
    img = fix_orientation(img)
    img = ensure_srgb(img).convert("RGB")
    context.user_data["bg_img"] = img

    blur_var = detect_blur(img)
    if blur_var < BLUR_THRESHOLD:
        await update.message.reply_text(_("blur_warn", context))

    await update.message.reply_text(_("color", context))
    return States.BG_WAIT_COLOR

async def bg_wait_color(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    color_text = update.message.text.strip()
    img = context.user_data.get("bg_img")
    if not img:
        await update.message.reply_text(_("error", context))
        release_lock(context)
        return ConversationHandler.END

    try:
        await update.message.reply_text(_("processing", context))
        result_img = simple_background_replace(img, color_text)
        context.user_data["bg_result"] = result_img
    except Exception as e:
        logger.error(f"Background change error: {e}")
        await update.message.reply_text(_("error", context) + f" {str(e)}")
        release_lock(context)
        return ConversationHandler.END

    preview = create_preview(result_img.copy())
    keyboard = [
        [InlineKeyboardButton(_("looks_ok", context), callback_data="bg_confirm")],
        [InlineKeyboardButton(_("change_settings", context), callback_data="bg_restart")],
    ]
    await update.message.reply_photo(
        photo=preview,
        caption=_("preview", context),
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return States.BG_PREVIEW

async def bg_preview_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "bg_confirm":
        keyboard = [
            [InlineKeyboardButton("JPEG", callback_data="JPEG"),
             InlineKeyboardButton("PNG", callback_data="PNG")],
            [InlineKeyboardButton("PDF", callback_data="PDF")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_caption(
            caption=_("format_choose", context),
            reply_markup=reply_markup
        )
        return States.BG_WAIT_FORMAT
    else:  # bg_restart
        await query.message.delete()
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=_("bg_change", context) + " " + _("send_photo", context)
        )
        context.user_data.pop("bg_img", None)
        context.user_data.pop("bg_result", None)
        return States.BG_WAIT_PHOTO

async def bg_wait_format(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    fmt = query.data

    result_img = context.user_data.get("bg_result")
    if not result_img:
        await query.edit_message_text(_("error", context))
        release_lock(context)
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
            result_img.save(output, format="PDF", resolution=dpi_val)
        output.seek(0)

        output_size = output.getbuffer().nbytes
        dimensions = f"{result_img.width} x {result_img.height} px"
        caption = (f"✅ Background changed! {VERSION}\n"
                   f"📏 {dimensions}\n"
                   f"📦 {format_file_size(output_size)}\n"
                   f"🖼️ Format: {fmt}\n"
                   f"{_('reminder', context)}")

        await query.message.reply_photo(
            photo=output,
            filename=filename,
            caption=caption
        )
    except Exception as e:
        logger.error(f"Format conversion error: {e}")
        await query.message.reply_text(_("error", context))
    finally:
        context.user_data.pop("bg_original", None)
        context.user_data.pop("bg_img", None)
        context.user_data.pop("bg_result", None)
        release_lock(context)

    return ConversationHandler.END

# ========================== CUSTOM SIZE FLOW ==========================
async def custom_wait_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    photo_file = await update.message.photo[-1].get_file()
    image_bytes = io.BytesIO()
    await photo_file.download_to_memory(image_bytes)
    image_bytes.seek(0)
    context.user_data["custom_original_bytes"] = image_bytes

    img = Image.open(image_bytes)
    img = fix_orientation(img)
    img = ensure_srgb(img)
    context.user_data["custom_original_img"] = img

    blur_var = detect_blur(img)
    if blur_var < BLUR_THRESHOLD:
        await update.message.reply_text(_("blur_warn", context))

    img = auto_level(img)
    context.user_data["custom_original_img"] = img

    await update.message.reply_text(_("dimensions", context))
    return States.CUSTOM_WAIT_DIMENSIONS

async def custom_wait_dimensions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.strip()
    try:
        w, h, unit = parse_dimensions(text)
        target_w, target_h = dimensions_to_pixels(w, h, unit)
        context.user_data["custom_target_w"] = target_w
        context.user_data["custom_target_h"] = target_h
    except ValueError as e:
        await update.message.reply_text(f"{_('error', context)}: {e}\n{_('dimensions', context)}")
        return States.CUSTOM_WAIT_DIMENSIONS

    img = context.user_data.get("custom_original_img")
    if img:
        orig_w, orig_h = img.size
        if not check_aspect_ratio(orig_w, orig_h, target_w, target_h):
            keyboard = [
                [InlineKeyboardButton("✅ Proceed anyway", callback_data="aspect_proceed")],
                [InlineKeyboardButton("🔄 Enter new dimensions", callback_data="aspect_retry")],
            ]
            await update.message.reply_text(
                "⚠️ The target aspect ratio differs significantly from the original. "
                "This may result in distortion or cropping. Do you want to continue?",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return States.CONFIRM_ASPECT

    return await ask_size_option(update, context)

async def confirm_aspect(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "aspect_proceed":
        await query.edit_message_text("Proceeding with current dimensions.")
        return await ask_size_option(update, context)
    else:
        await query.edit_message_text(_("dimensions", context))
        return States.CUSTOM_WAIT_DIMENSIONS

async def ask_size_option(update, context):
    if update.callback_query:
        query = update.callback_query
        await query.answer()
        msg = query.edit_message_text
    else:
        msg = update.message.reply_text

    keyboard = [
        [InlineKeyboardButton("Yes, specify file size", callback_data="size_yes")],
        [InlineKeyboardButton("No, just resize", callback_data="size_no")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await msg(
        "Do you want to set a maximum file size (e.g., 50 KB)?",
        reply_markup=reply_markup,
    )
    return States.CUSTOM_WAIT_SIZE_OPTION

async def custom_wait_size_option(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "size_yes":
        await query.edit_message_text(_("size_target", context))
        return States.CUSTOM_WAIT_TARGET_SIZE
    else:
        context.user_data["custom_target_kb"] = None
        keyboard = [
            [InlineKeyboardButton("JPEG", callback_data="JPEG"),
             InlineKeyboardButton("PNG", callback_data="PNG")],
            [InlineKeyboardButton("PDF", callback_data="PDF")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(_("format_choose", context), reply_markup=reply_markup)
        return States.CUSTOM_WAIT_FORMAT

async def custom_wait_target_size(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        target_kb = int(update.message.text.strip())
        context.user_data["custom_target_kb"] = target_kb
    except ValueError:
        await update.message.reply_text(_("error", context) + " " + _("size_target", context))
        return States.CUSTOM_WAIT_TARGET_SIZE

    keyboard = [
        [InlineKeyboardButton("JPEG", callback_data="JPEG"),
         InlineKeyboardButton("PNG", callback_data="PNG")],
        [InlineKeyboardButton("PDF", callback_data="PDF")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(_("format_choose", context), reply_markup=reply_markup)
    return States.CUSTOM_WAIT_FORMAT

async def custom_wait_format(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    fmt = query.data

    img = context.user_data.get("custom_original_img")
    target_w = context.user_data.get("custom_target_w")
    target_h = context.user_data.get("custom_target_h")
    target_kb = context.user_data.get("custom_target_kb")
    strict = context.user_data.get("strict", False)
    dpi_val = context.user_data.get("dpi", DPI)

    if not all([img, target_w, target_h]):
        await query.edit_message_text(_("error", context))
        release_lock(context)
        return ConversationHandler.END

    try:
        await query.message.reply_text(_("processing", context))
        processed_img = center_face_in_dimensions(img, target_w, target_h)

        if target_kb is not None:
            if fmt == "JPEG":
                output = compress_to_target_size(processed_img, target_kb, fmt, strict)
                final_size = output.getbuffer().nbytes
                if final_size > (target_kb * 1024) * (1.01 if strict else 1.05):
                    raise Exception(f"Could not compress to {target_kb} KB (got {final_size/1024:.1f} KB).")
                context.user_data["custom_final"] = output
                context.user_data["custom_final_caption"] = f"Resized & centered\n📏 {target_w} x {target_h} px\n📦 {format_file_size(final_size)}\n🖼️ {fmt}"
                context.user_data["custom_final_filename"] = f"photo_ready_{VERSION}.{fmt.lower()}"
            else:
                output = compress_to_target_size(processed_img, target_kb, fmt, strict)
                final_size = output.getbuffer().nbytes
                if final_size > (target_kb * 1024) * (1.01 if strict else 1.05):
                    context.user_data["pending_output"] = output
                    context.user_data["pending_caption"] = f"Resized & centered\n📏 {target_w} x {target_h} px\n📦 {format_file_size(final_size)} (exceeds {target_kb} KB)\n🖼️ {fmt}"
                    context.user_data["pending_filename"] = f"photo_ready_{VERSION}.{fmt.lower()}"
                    keyboard = [[InlineKeyboardButton("✅ Send anyway", callback_data="confirm_send"),
                                 InlineKeyboardButton("❌ Cancel", callback_data="cancel_send")]]
                    await query.message.reply_text(
                        f"⚠️ The output file size is {format_file_size(final_size)}, which exceeds your target of {target_kb} KB. Do you still want to send it?",
                        reply_markup=InlineKeyboardMarkup(keyboard)
                    )
                    return States.CONFIRM_OVERSIZE
                else:
                    context.user_data["custom_final"] = output
                    context.user_data["custom_final_caption"] = f"Resized & centered\n📏 {target_w} x {target_h} px\n📦 {format_file_size(final_size)}\n🖼️ {fmt}"
                    context.user_data["custom_final_filename"] = f"photo_ready_{VERSION}.{fmt.lower()}"
        else:
            output = io.BytesIO()
            if fmt == "JPEG":
                processed_img.save(output, format="JPEG", quality=95, dpi=(dpi_val, dpi_val))
            elif fmt == "PNG":
                pnginfo = Image.PngImageInfo()
                pnginfo.dpi = (dpi_val, dpi_val)
                processed_img.save(output, format="PNG", pnginfo=pnginfo)
            elif fmt == "PDF":
                processed_img.save(output, format="PDF", resolution=dpi_val)
            output.seek(0)
            final_size = output.getbuffer().nbytes
            context.user_data["custom_final"] = output
            context.user_data["custom_final_caption"] = f"Resized & centered\n📏 {target_w} x {target_h} px\n📦 {format_file_size(final_size)}\n🖼️ {fmt}"
            context.user_data["custom_final_filename"] = f"photo_ready_{VERSION}.{fmt.lower()}"

        preview = create_preview(processed_img.copy())
        keyboard = [
            [InlineKeyboardButton(_("looks_ok", context), callback_data="custom_confirm")],
            [InlineKeyboardButton(_("change_settings", context), callback_data="custom_restart")],
        ]
        await query.message.reply_photo(
            photo=preview,
            caption=_("preview", context),
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return States.CUSTOM_PREVIEW

    except Exception as e:
        logger.error(f"Custom resize error: {e}")
        await query.message.reply_text(_("error", context) + f" {str(e)}")
        release_lock(context)
        return ConversationHandler.END

async def custom_preview_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "custom_confirm":
        final_output = context.user_data.get("custom_final")
        caption = context.user_data.get("custom_final_caption")
        filename = context.user_data.get("custom_final_filename")
        if final_output and caption and filename:
            await query.message.reply_photo(
                photo=final_output,
                filename=filename,
                caption="✅ " + caption + "\n" + _("reminder", context)
            )
        else:
            await query.message.reply_text(_("error", context))
    else:  # custom_restart
        await query.message.delete()
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=_("resize", context) + " " + _("send_photo", context)
        )
        for key in ["custom_original_img", "custom_target_w", "custom_target_h", "custom_target_kb", "custom_final"]:
            context.user_data.pop(key, None)
        return States.CUSTOM_WAIT_PHOTO

    for key in ["custom_original_bytes", "custom_original_img", "custom_target_w", "custom_target_h", "custom_target_kb", "custom_final", "custom_final_caption", "custom_final_filename"]:
        context.user_data.pop(key, None)
    release_lock(context)
    return ConversationHandler.END

# ========================== REDUCE SIZE ONLY FLOW ==========================
async def reduce_wait_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    photo_file = await update.message.photo[-1].get_file()
    image_bytes = io.BytesIO()
    await photo_file.download_to_memory(image_bytes)
    image_bytes.seek(0)
    context.user_data["reduce_original_bytes"] = image_bytes

    img = Image.open(image_bytes)
    img = fix_orientation(img)
    img = ensure_srgb(img)
    context.user_data["reduce_original_img"] = img

    blur_var = detect_blur(img)
    if blur_var < BLUR_THRESHOLD:
        await update.message.reply_text(_("blur_warn", context))

    await update.message.reply_text(_("size_target", context))
    return States.REDUCE_WAIT_TARGET_SIZE

async def reduce_wait_target_size(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        target_kb = int(update.message.text.strip())
        context.user_data["reduce_target_kb"] = target_kb
    except ValueError:
        await update.message.reply_text(_("error", context) + " " + _("size_target", context))
        return States.REDUCE_WAIT_TARGET_SIZE

    keyboard = [
        [InlineKeyboardButton("JPEG", callback_data="JPEG"),
         InlineKeyboardButton("PNG", callback_data="PNG")],
        [InlineKeyboardButton("PDF", callback_data="PDF")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        _("format_choose", context) + " (JPEG recommended)",
        reply_markup=reply_markup,
    )
    return States.REDUCE_WAIT_FORMAT

async def reduce_wait_format(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    fmt = query.data

    img = context.user_data.get("reduce_original_img")
    target_kb = context.user_data.get("reduce_target_kb")
    strict = context.user_data.get("strict", False)
    dpi_val = context.user_data.get("dpi", DPI)

    if not img or not target_kb:
        await query.edit_message_text(_("error", context))
        release_lock(context)
        return ConversationHandler.END

    try:
        await query.message.reply_text(_("processing", context))
        if fmt == "JPEG":
            output = compress_to_target_size(img, target_kb, fmt, strict)
            final_size = output.getbuffer().nbytes
            if final_size > (target_kb * 1024) * (1.01 if strict else 1.05):
                raise Exception(f"Could not compress to {target_kb} KB (got {final_size/1024:.1f} KB).")
            context.user_data["reduce_final"] = output
            context.user_data["reduce_final_caption"] = f"Compressed\n📏 {img.width} x {img.height} px\n📦 {format_file_size(final_size)}\n🖼️ {fmt}"
            context.user_data["reduce_final_filename"] = f"compressed_{VERSION}.{fmt.lower()}"
        else:
            output = compress_to_target_size(img, target_kb, fmt, strict)
            final_size = output.getbuffer().nbytes
            if final_size > (target_kb * 1024) * (1.01 if strict else 1.05):
                context.user_data["pending_output"] = output
                context.user_data["pending_caption"] = f"Compressed\n📏 {img.width} x {img.height} px\n📦 {format_file_size(final_size)} (exceeds {target_kb} KB)\n🖼️ {fmt}"
                context.user_data["pending_filename"] = f"compressed_{VERSION}.{fmt.lower()}"
                keyboard = [[InlineKeyboardButton("✅ Send anyway", callback_data="confirm_send"),
                             InlineKeyboardButton("❌ Cancel", callback_data="cancel_send")]]
                await query.message.reply_text(
                    f"⚠️ The output file size is {format_file_size(final_size)}, which exceeds your target of {target_kb} KB. Do you still want to send it?",
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
                return States.CONFIRM_OVERSIZE
            else:
                context.user_data["reduce_final"] = output
                context.user_data["reduce_final_caption"] = f"Compressed\n📏 {img.width} x {img.height} px\n📦 {format_file_size(final_size)}\n🖼️ {fmt}"
                context.user_data["reduce_final_filename"] = f"compressed_{VERSION}.{fmt.lower()}"

        preview_img = Image.open(io.BytesIO(output.getvalue())) if fmt != "PDF" else img
        preview = create_preview(preview_img.copy())
        keyboard = [
            [InlineKeyboardButton(_("looks_ok", context), callback_data="reduce_confirm")],
            [InlineKeyboardButton(_("change_settings", context), callback_data="reduce_restart")],
        ]
        await query.message.reply_photo(
            photo=preview,
            caption=_("preview", context),
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return States.REDUCE_PREVIEW

    except Exception as e:
        logger.error(f"Reduce size error: {e}")
        await query.message.reply_text(_("error", context) + f" {str(e)}")
        release_lock(context)
        return ConversationHandler.END

async def reduce_preview_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "reduce_confirm":
        final_output = context.user_data.get("reduce_final")
        caption = context.user_data.get("reduce_final_caption")
        filename = context.user_data.get("reduce_final_filename")
        if final_output and caption and filename:
            await query.message.reply_photo(
                photo=final_output,
                filename=filename,
                caption="✅ " + caption + "\n" + _("reminder", context)
            )
        else:
            await query.message.reply_text(_("error", context))
    else:  # reduce_restart
        await query.message.delete()
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=_("send_photo", context)
        )
        for key in ["reduce_original_img", "reduce_target_kb", "reduce_final"]:
            context.user_data.pop(key, None)
        return States.REDUCE_WAIT_PHOTO

    for key in ["reduce_original_bytes", "reduce_original_img", "reduce_target_kb", "reduce_final", "reduce_final_caption", "reduce_final_filename"]:
        context.user_data.pop(key, None)
    release_lock(context)
    return ConversationHandler.END

# ========================== SIGNATURE FLOW ==========================
async def signature_wait_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    photo_file = await update.message.photo[-1].get_file()
    image_bytes = io.BytesIO()
    await photo_file.download_to_memory(image_bytes)
    image_bytes.seek(0)
    context.user_data["sig_original_bytes"] = image_bytes

    img = Image.open(image_bytes)
    img = fix_orientation(img)
    img = ensure_srgb(img)
    context.user_data["sig_original_img"] = img

    if not check_signature_bg(img):
        await update.message.reply_text(_("signature_bg_warn", context))

    await update.message.reply_text(
        "Please enter the desired signature height in pixels (e.g., 50).\n"
        "Width will be adjusted automatically."
    )
    return States.SIGNATURE_WAIT_HEIGHT

async def signature_wait_height(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        height = int(update.message.text.strip())
        context.user_data["sig_height"] = height
    except ValueError:
        await update.message.reply_text("Please enter a valid number (e.g., 50).")
        return States.SIGNATURE_WAIT_HEIGHT

    await update.message.reply_text(_("size_target", context))
    return States.SIGNATURE_WAIT_SIZE

async def signature_wait_size(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        target_kb = int(update.message.text.strip())
        context.user_data["sig_target_kb"] = target_kb
    except ValueError:
        await update.message.reply_text(_("error", context) + " " + _("size_target", context))
        return States.SIGNATURE_WAIT_SIZE

    keyboard = [
        [InlineKeyboardButton("JPEG", callback_data="JPEG"),
         InlineKeyboardButton("PNG", callback_data="PNG")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        _("format_choose", context) + " (JPEG recommended)",
        reply_markup=reply_markup,
    )
    return States.SIGNATURE_WAIT_FORMAT

async def signature_wait_format(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    fmt = query.data

    img = context.user_data.get("sig_original_img")
    height = context.user_data.get("sig_height")
    target_kb = context.user_data.get("sig_target_kb")
    strict = context.user_data.get("strict", False)
    dpi_val = context.user_data.get("dpi", DPI)

    if not all([img, height, target_kb]):
        await query.edit_message_text(_("error", context))
        release_lock(context)
        return ConversationHandler.END

    try:
        await query.message.reply_text(_("processing", context))
        output = process_signature(img, height, target_kb, strict)
        final_size = output.getbuffer().nbytes
        if final_size > (target_kb * 1024) * (1.01 if strict else 1.05) and fmt == "JPEG":
            raise Exception(f"Could not compress to {target_kb} KB (got {final_size/1024:.1f} KB).")
        elif final_size > (target_kb * 1024) * (1.01 if strict else 1.05) and fmt == "PNG":
            context.user_data["pending_output"] = output
            context.user_data["pending_caption"] = f"Signature\n📏 (after resize) {img.width} x {img.height} px\n📦 {format_file_size(final_size)} (exceeds {target_kb} KB)\n🖼️ {fmt}"
            context.user_data["pending_filename"] = f"signature_{VERSION}.{fmt.lower()}"
            keyboard = [[InlineKeyboardButton("✅ Send anyway", callback_data="confirm_send"),
                         InlineKeyboardButton("❌ Cancel", callback_data="cancel_send")]]
            await query.message.reply_text(
                f"⚠️ The output file size is {format_file_size(final_size)}, which exceeds your target of {target_kb} KB. Do you still want to send it?",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return States.CONFIRM_OVERSIZE

        context.user_data["sig_final"] = output
        context.user_data["sig_final_caption"] = f"Signature ready\n📏 (after resize) {img.width} x {img.height} px\n📦 {format_file_size(final_size)}\n🖼️ {fmt}"
        context.user_data["sig_final_filename"] = f"signature_{VERSION}.{fmt.lower()}"

        preview_img = Image.open(io.BytesIO(output.getvalue()))
        preview = create_preview(preview_img.copy())
        keyboard = [
            [InlineKeyboardButton(_("looks_ok", context), callback_data="sig_confirm")],
            [InlineKeyboardButton(_("change_settings", context), callback_data="sig_restart")],
        ]
        await query.message.reply_photo(
            photo=preview,
            caption=_("preview", context),
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return States.SIGNATURE_PREVIEW

    except Exception as e:
        logger.error(f"Signature error: {e}")
        await query.message.reply_text(_("error", context) + f" {str(e)}")
        release_lock(context)
        return ConversationHandler.END

async def signature_preview_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "sig_confirm":
        final_output = context.user_data.get("sig_final")
        caption = context.user_data.get("sig_final_caption")
        filename = context.user_data.get("sig_final_filename")
        if final_output and caption and filename:
            await query.message.reply_photo(
                photo=final_output,
                filename=filename,
                caption="✅ " + caption + "\n" + _("reminder", context)
            )
        else:
            await query.message.reply_text(_("error", context))
    else:  # sig_restart
        await query.message.delete()
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=_("signature", context) + " " + _("send_photo", context)
        )
        for key in ["sig_original_img", "sig_height", "sig_target_kb", "sig_final"]:
            context.user_data.pop(key, None)
        return States.SIGNATURE_WAIT_PHOTO

    for key in ["sig_original_bytes", "sig_original_img", "sig_height", "sig_target_kb", "sig_final", "sig_final_caption", "sig_final_filename"]:
        context.user_data.pop(key, None)
    release_lock(context)
    return ConversationHandler.END

# ========================== CONFIRM OVERSIZE HANDLER ==========================
async def confirm_oversize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "confirm_send":
        output = context.user_data.get("pending_output")
        caption = context.user_data.get("pending_caption")
        filename = context.user_data.get("pending_filename")
        if output and caption and filename:
            await query.message.reply_photo(
                photo=output,
                filename=filename,
                caption="⚠️ " + caption + "\n" + _("reminder", context)
            )
        else:
            await query.message.reply_text(_("error", context))
    else:
        await query.message.reply_text(_("cancel", context))

    context.user_data.pop("pending_output", None)
    context.user_data.pop("pending_caption", None)
    context.user_data.pop("pending_filename", None)
    release_lock(context)
    return ConversationHandler.END

# ========================== CANCEL AND FALLBACK ==========================
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(_("cancel", context))
    context.user_data.clear()
    return ConversationHandler.END

async def fallback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send the main menu with buttons when the bot is not in a conversation."""
    keyboard = [
        [InlineKeyboardButton(_("bg_change", context), callback_data="bg_change")],
        [InlineKeyboardButton(_("resize", context), callback_data="resize")],
        [InlineKeyboardButton(_("signature", context), callback_data="signature")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        _("start", context),
        reply_markup=reply_markup,
    )

# ========================== MAIN ==========================
def main():
    token = os.environ.get("BOT_TOKEN")
    if not token:
        raise ValueError("No BOT_TOKEN environment variable set")

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("strict", strict_mode))
    application.add_handler(CommandHandler("hinglish", hinglish_mode))
    application.add_handler(CommandHandler("dpi", set_dpi_command))

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            States.SELECT_ACTION: [CallbackQueryHandler(select_action)],
            States.BG_WAIT_PHOTO: [MessageHandler(filters.PHOTO, bg_wait_photo)],
            States.BG_WAIT_COLOR: [MessageHandler(filters.TEXT & ~filters.COMMAND, bg_wait_color)],
            States.BG_PREVIEW: [CallbackQueryHandler(bg_preview_confirm)],
            States.BG_WAIT_FORMAT: [CallbackQueryHandler(bg_wait_format)],
            States.RESIZE_MODE: [CallbackQueryHandler(resize_mode)],
            States.CUSTOM_WAIT_PHOTO: [MessageHandler(filters.PHOTO, custom_wait_photo)],
            States.CUSTOM_WAIT_DIMENSIONS: [MessageHandler(filters.TEXT & ~filters.COMMAND, custom_wait_dimensions)],
            States.CUSTOM_WAIT_SIZE_OPTION: [CallbackQueryHandler(custom_wait_size_option)],
            States.CUSTOM_WAIT_TARGET_SIZE: [MessageHandler(filters.TEXT & ~filters.COMMAND, custom_wait_target_size)],
            States.CUSTOM_WAIT_FORMAT: [CallbackQueryHandler(custom_wait_format)],
            States.CUSTOM_PREVIEW: [CallbackQueryHandler(custom_preview_confirm)],
            States.REDUCE_WAIT_PHOTO: [MessageHandler(filters.PHOTO, reduce_wait_photo)],
            States.REDUCE_WAIT_TARGET_SIZE: [MessageHandler(filters.TEXT & ~filters.COMMAND, reduce_wait_target_size)],
            States.REDUCE_WAIT_FORMAT: [CallbackQueryHandler(reduce_wait_format)],
            States.REDUCE_PREVIEW: [CallbackQueryHandler(reduce_preview_confirm)],
            States.SIGNATURE_WAIT_PHOTO: [MessageHandler(filters.PHOTO, signature_wait_photo)],
            States.SIGNATURE_WAIT_HEIGHT: [MessageHandler(filters.TEXT & ~filters.COMMAND, signature_wait_height)],
            States.SIGNATURE_WAIT_SIZE: [MessageHandler(filters.TEXT & ~filters.COMMAND, signature_wait_size)],
            States.SIGNATURE_WAIT_FORMAT: [CallbackQueryHandler(signature_wait_format)],
            States.SIGNATURE_PREVIEW: [CallbackQueryHandler(signature_preview_confirm)],
            States.CONFIRM_OVERSIZE: [CallbackQueryHandler(confirm_oversize)],
            States.CONFIRM_ASPECT: [CallbackQueryHandler(confirm_aspect)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv_handler)
    application.add_handler(MessageHandler(filters.ALL, fallback))

    application.run_polling()

if __name__ == "__main__":
    main()
