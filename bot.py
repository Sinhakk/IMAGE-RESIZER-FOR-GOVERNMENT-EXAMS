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
    "reminder": "⚠️ Form upload se pehle preview check kar lena. Ye tool sirf madad ke liye hai, guarantee nahi deta.",
    "bg_warning": "⚠️ Background change mein kami reh sakti hai. Preview dhyan se dekhein.",
    "main_menu": "Main menu:",
    "unexpected": "Unexpected input. Please follow the instructions above.",
    "size_too_large": "Photo ka size bahut bada hai. Chhota kar rahe hain...",
    "rate_limit": "Aapne bahut jyada requests bhej di hain. Thodi der ruk kar try karein.",
    "timeout_error": "Processing mein zyada time lag raha hai. Kam resolution ki photo use karein.",
    "face_small_warning": "Face chhota hai. Better photo use karein.",
    "face_offcenter_warning": "Face centre mein nahi hai. Better framing karein.",
    "background_presets": "Background choose karein:",
    "history": "Last processed image dubara bhejna hai?",
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
    "reminder": "⚠️ Please double-check the output before uploading. This tool is only for assistance, no guarantee.",
    "bg_warning": "⚠️ Background change may have imperfections. Check preview carefully.",
    "main_menu": "Main menu:",
    "unexpected": "Unexpected input. Please follow the instructions above.",
    "size_too_large": "Image is too large. Downsizing for processing...",
    "rate_limit": "You have sent too many requests. Please wait a while.",
    "timeout_error": "Processing is taking too long. Try with a lower resolution image.",
    "face_small_warning": "Face is too small. Use a better photo.",
    "face_offcenter_warning": "Face is not centered. Improve framing.",
    "background_presets": "Choose a background:",
    "history": "Resend last processed image?",
}

def _(key, context):
    # Safety check: ensure dictionaries are actually dicts
    hinglish_dict = LANG_HINGLISH if isinstance(LANG_HINGLISH, dict) else {}
    english_dict = LANG_EN if isinstance(LANG_EN, dict) else {}
    if context.user_data.get("hinglish", False):
        return hinglish_dict.get(key, english_dict.get(key, key))
    return english_dict.get(key, key)
