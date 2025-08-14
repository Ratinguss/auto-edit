import os, math, tempfile, logging
from typing import Callable, Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont  # add (Image already imported above for ANTIALIAS monkey patch)

logging.basicConfig(level=logging.INFO)

# Monkey patch for Pillow 10+ (MoviePy expects Image.ANTIALIAS)
from PIL import Image
if not hasattr(Image, "ANTIALIAS"):
    from PIL import Image as _Img
    Image.ANTIALIAS = Image.Resampling.LANCZOS  # type: ignore

import numpy as np  # ok if installed; else wrap in try if needed

from moviepy.editor import (
    VideoFileClip,
    concatenate_videoclips,
    CompositeVideoClip,
    CompositeAudioClip,
    AudioFileClip,
    ImageClip,
    TextClip,
    ColorClip,
)
from moviepy.video.fx import all as vfx

try:
    from pydub import AudioSegment
    from pydub.silence import detect_silence
except ImportError:
    AudioSegment = None
    detect_silence = None

ProgressFn = Callable[[int, str], None]

# ---------------- File discovery ----------------
def discover_job_files(job_id: str, upload_folder: str) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    prefix = f"{job_id}_"
    for name in os.listdir(upload_folder):
        if not name.startswith(prefix):
            continue
        parts = name.split("_", 2)
        if len(parts) >= 3:
            _, field, original = parts
            mapping.setdefault(field, []).append(os.path.join(upload_folder, name))
    for k in mapping:
        mapping[k].sort()
    return mapping

# ---------------- Silence removal (pydub) ----------------
# Tunable defaults (make less aggressive than previous 200ms/-45dB)
SILENCE_MIN_LEN_MS_DEFAULT = 600      # only cut pauses >= 0.6s
SILENCE_THRESH_DB_DEFAULT  = -50      # treat only very quiet parts as silence
SILENCE_PADDING_MS_DEFAULT = 120      # keep some context around spoken parts
SILENCE_MAX_REMOVAL_RATIO  = 0.40     # if more than 40% would be removed, abort trimming

def remove_silence_from_clip(
    clip: VideoFileClip,
    min_silence_len_ms: int = SILENCE_MIN_LEN_MS_DEFAULT,
    silence_thresh_db: int = SILENCE_THRESH_DB_DEFAULT,
    padding_ms: int = SILENCE_PADDING_MS_DEFAULT,
    max_removal_ratio: float = SILENCE_MAX_REMOVAL_RATIO,
) -> VideoFileClip:
    """Detect and remove long silences from a clip.

    Parameters:
        min_silence_len_ms: minimum contiguous silence to cut (ms).
        silence_thresh_db: anything below this dBFS (negative) considered silence.
        padding_ms: how much audio (ms) to retain before & after each kept region.
        max_removal_ratio: safety cap; abort if proposed removal > ratio of total.

    To make trimming LESS aggressive increase min_silence_len_ms or make silence_thresh_db MORE negative (e.g. -55),
    or set padding_ms higher. To disable entirely, skip calling this function.
    """
    if AudioSegment is None or detect_silence is None or clip.audio is None:
        return clip
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        clip.audio.write_audiofile(tmp_path, verbose=False, logger=None)
        audio_seg = AudioSegment.from_file(tmp_path)
        os.remove(tmp_path)

        silences = detect_silence(
            audio_seg,
            min_silence_len=min_silence_len_ms,
            silence_thresh=silence_thresh_db,
            seek_step=10,
        )
        if not silences:
            return clip

        duration_ms = len(audio_seg)
        # Invert silence regions to speech keep regions
        keep_regions: List[Tuple[int,int]] = []
        cursor = 0
        for start, end in silences:
            if start > cursor:
                keep_regions.append((cursor, start))
            cursor = end
        if cursor < duration_ms:
            keep_regions.append((cursor, duration_ms))

        if not keep_regions:
            return clip

        # Apply padding around speech regions and clamp
        padded: List[Tuple[int,int]] = []
        for a,b in keep_regions:
            a2 = max(0, a - padding_ms)
            b2 = min(duration_ms, b + padding_ms)
            if not padded:
                padded.append([a2,b2])
            else:
                prev = padded[-1]
                if a2 - prev[1] < 80:  # merge if overlaps / near
                    prev[1] = max(prev[1], b2)
                else:
                    padded.append([a2,b2])
        keep_regions = [(a,b) for a,b in padded if (b-a) > 50]  # drop micro segments

        # Safety: estimate removal ratio
        kept_ms = sum(b-a for a,b in keep_regions)
        removal_ratio = 1 - (kept_ms / duration_ms)
        if removal_ratio > max_removal_ratio:
            logging.info(f"Silence trim skipped (would remove {removal_ratio:.1%} > {max_removal_ratio:.0%})")
            return clip

        subclips = [clip.subclip(a/1000.0, b/1000.0) for a,b in keep_regions]
        if len(subclips) == 1:
            return subclips[0]
        joined = concatenate_videoclips(subclips, method="compose")
        for sc in subclips:
            if sc != joined:
                try: sc.close()
                except: pass
        return joined
    except Exception as e:
        logging.warning(f"Silence removal failed: {e}")
        return clip

# ---------------- Transcription ----------------
def transcribe_audio(path: str) -> List[Dict]:
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(path)
        segs = []
        for seg in result.get("segments", []):
            segs.append(
                {
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "text": seg.get("text", "").strip(),
                }
            )
        return segs
    except Exception as e:
        logging.warning(f"Transcription failed: {e}")
        return []

# ---------------- Captions ----------------
TARGET_SIZE = (720, 1280)
BLUR_RADIUS = 15
CAPTION_FONT_SIZE = 42
CAPTION_MAX_GROUP_DURATION = 2.0
CAPTION_MAX_GAP = 0.5
# New limits to shorten each on-screen caption
CAPTION_MAX_WORDS = 7          # max words per caption chunk
CAPTION_MAX_CHARS = 42         # max characters per chunk (soft cap)
CAPTION_MIN_CHUNK_SEC = 0.6    # min display time per chunk (advisory)

# ---- B-roll display config ----
BROLL_MODE = "stretch"   # options: cover, contain, contain_blur, preserve_if_vertical, stretch
BROLL_MAX_UPSCALE = 1.15
AUTO_ROTATE_BROLL = True
BROLL_MIN_TOKEN_LEN = 3
LOG_BROLL_MATCH = True
# New B-roll rules
BROLL_TRIGGER_EARLY_SEC = 0.2   # start 0.2s before matched word
BROLL_FIXED_DUR = 4.0           # fixed display duration (seconds)
BROLL_UNIQUE = True             # use each file at most once

# New Overlay rules
OVERLAY_TRIGGER_EARLY_SEC = 0.2
OVERLAY_FIXED_DUR = 3.0
OVERLAY_UNIQUE = True

def _group_segments(segments: List[Dict],
                    max_gap: float = CAPTION_MAX_GAP,
                    max_duration: float = CAPTION_MAX_GROUP_DURATION) -> List[Dict]:
    groups = []
    if not segments: return groups
    cur = {"start": segments[0]["start"], "end": segments[0]["end"], "texts": [segments[0]["text"]]}
    for seg in segments[1:]:
        gap = seg["start"] - cur["end"]
        dur = seg["end"] - cur["start"]
        if gap <= max_gap and dur <= max_duration:
            cur["end"] = seg["end"]
            cur["texts"].append(seg["text"])
        else:
            groups.append(cur)
            cur = {"start": seg["start"], "end": seg["end"], "texts": [seg["text"]]}
    groups.append(cur)
    for g in groups:
        g["text"] = " ".join(t.strip() for t in g["texts"] if t.strip())
        del g["texts"]
    return groups

def build_caption_clips(
    segments: List[Dict],
    video_size: Tuple[int, int],
    font_size_px: int = CAPTION_FONT_SIZE,
) -> List[VideoFileClip]:
    """
    Tries MoviePy TextClip first; falls back to PIL-rendered ImageClips if TextClip fails.
    """
    CAPTION_VERTICAL_FRACTION = 0.25
    if not segments:
        logging.info("No caption segments to render.")
        return []
    grouped = _group_segments(segments)
    logging.info(f"Caption grouping: {len(segments)} -> {len(grouped)} groups")
    W, H = video_size

    bottom_margin = int(H * CAPTION_VERTICAL_FRACTION)
    clips: List[VideoFileClip] = []

    try:
        pil_font = ImageFont.truetype("arial.ttf", font_size_px)
    except Exception:
        pil_font = ImageFont.load_default()

    # Helper: split long text into chunks by words/chars
    def chunk_text_by_limits(text: str) -> List[str]:
        words = text.split()
        if not words:
            return []
        chunks, cur = [], []
        cur_len = 0
        for w in words:
            add_len = (1 if cur else 0) + len(w)
            if (len(cur) + 1 > CAPTION_MAX_WORDS) or (cur_len + add_len > CAPTION_MAX_CHARS):
                chunks.append(" ".join(cur))
                cur, cur_len = [w], len(w)
            else:
                cur.append(w)
                cur_len += add_len
        if cur:
            chunks.append(" ".join(cur))
        return chunks

    for g in grouped:
        text = g["text"]
        if not text: 
            continue
        start = g["start"]
        duration = max(0.01, g["end"] - g["start"])

        chunks = chunk_text_by_limits(text)
        if not chunks:
            chunks = [text]
        n = len(chunks)

        for i, chunk in enumerate(chunks):
            # Evenly spread the group time over its chunks
            sub_start = start + (duration * i / n)
            sub_dur = max(0.01, duration / n)  # keep simple; respects original timing

            # Try TextClip first
            try:
                tc = (TextClip(
                        chunk,
                        fontsize=font_size_px,
                        font="Arial",
                        color="white",
                        method="caption",
                        size=(int(W * 0.9), None),
                        align="center"
                    ).set_start(sub_start).set_duration(sub_dur)
                     .set_position(("center", H - bottom_margin)))
                clips.append(tc)
                continue
            except Exception:
                pass

            # PIL fallback with centered lines
            try:
                words = chunk.split()
                lines = []
                current = ""
                max_chars = max(10, int(W * 0.9 / (font_size_px * 0.55)))
                for w in words:
                    if len(current) + (1 if current else 0) + len(w) <= max_chars:
                        current = (current + " " + w).strip()
                    else:
                        lines.append(current); current = w
                if current: lines.append(current)

                line_height = font_size_px + 4
                pad_y = 16
                text_h = line_height * len(lines)
                img_w = int(W * 0.9)
                img_h = text_h + pad_y * 2

                from PIL import Image as PILImage, ImageDraw as PILImageDraw
                img = PILImage.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
                draw = PILImageDraw.Draw(img)
                draw.rounded_rectangle([(0,0),(img_w,img_h)], radius=18, fill=(0,0,0,130))

                for li, line in enumerate(lines):
                    y = pad_y + li * line_height
                    try:
                        bbox = draw.textbbox((0,0), line, font=pil_font)
                        line_w = bbox[2] - bbox[0]
                    except Exception:
                        line_w = font_size_px * 0.55 * len(line)
                    x = (img_w - line_w) / 2
                    draw.text((x, y), line, font=pil_font, fill=(255,255,255,255))

                pil_clip = (ImageClip(np.array(img))
                            .set_start(sub_start).set_duration(sub_dur)
                            .set_position(("center", H - bottom_margin)))
                clips.append(pil_clip)
            except Exception as e2:
                logging.warning(f"PIL caption group failed: {e2}")

    return clips

# ---------------- Rounded image ----------------
def rounded_image_clip(path: str, duration: float, target_height: int, radius: int = 40):
    try:
        # Load and flatten to opaque RGB (remove any source alpha)
        from PIL import Image as PILImage, ImageDraw as PILImageDraw
        pil = PILImage.open(path).convert("RGBA")
        bg = PILImage.new("RGB", pil.size, (255, 255, 255))  # choose bg color if needed
        bg.paste(pil, mask=pil.split()[-1])  # composite removes original transparency
        img = ImageClip(np.array(bg)).set_duration(duration)
    except Exception:
        return ColorClip(size=(200, 200), color=(0, 0, 0)).set_duration(duration)

    # Resize/crop to your target aspect (kept as before)
    h = target_height
    w = int(h * 3 / 4)
    img = img.resize(height=h)
    if img.w > w:
        x1 = (img.w - w) // 2
        img = img.crop(x1=x1, y1=0, x2=x1 + w, y2=h)

    # Rounded corner mask (normalized 0..1)
    try:
        r = max(1, min(int(radius), min(img.w, img.h) // 2))
        mask_img = PILImage.new("L", (img.w, img.h), 0)
        draw = PILImageDraw.Draw(mask_img)
        draw.rounded_rectangle([(0, 0), (img.w, img.h)], radius=r, fill=255)
        mask_arr = (np.array(mask_img).astype("float32") / 255.0)
        mask_clip = ImageClip(mask_arr, ismask=True).set_duration(duration)
        return img.set_mask(mask_clip).set_opacity(1.0)
    except Exception:
        return img

# ---------------- Aspect / vertical ----------------
def adapt_clip_to_vertical(
    clip: VideoFileClip,
    target_size=TARGET_SIZE,
    blur_bg=True,
    blur_radius=BLUR_RADIUS
) -> VideoFileClip:
    Wt, Ht = target_size
    ar_target = Wt / Ht
    ar_clip = clip.w / clip.h
    if abs(ar_clip - ar_target) < 0.02:
        return clip.resize(newsize=target_size)
    if ar_clip < ar_target:
        scaled = clip.resize(height=Ht)
        bg = (
            scaled.resize((Wt, Ht)).fx(vfx.blur, blur_radius).set_duration(clip.duration)
            if blur_bg else ColorClip(size=(Wt, Ht), color=(0,0,0)).set_duration(clip.duration)
        )
        return CompositeVideoClip([bg, scaled.set_position("center")], size=(Wt, Ht))
    fg = clip.resize(height=Ht)
    bg = (
        fg.resize(width=Wt).fx(vfx.blur, blur_radius).set_duration(clip.duration)
        if blur_bg else ColorClip(size=(Wt, Ht), color=(0,0,0)).set_duration(clip.duration)
    )
    return CompositeVideoClip([bg, fg.set_position("center")], size=(Wt, Ht))

# ---------------- Audio normalize ----------------
def normalize_audio_level(clip: VideoFileClip, target_peak: float = 0.9) -> VideoFileClip:
    if clip.audio is None or np is None:
        return clip
    try:
        arr = clip.audio.to_soundarray(fps=44100)
        peak = float(np.max(np.abs(arr)))
        if peak <= 0:
            return clip
        factor = min(target_peak / peak, 10.0)
        return clip.volumex(factor)
    except Exception as e:
        logging.debug(f"Normalize failed: {e}")
        return clip

# ---------------- Dynamic Zoom ----------------
ZOOM_IN_START = 1.00
ZOOM_IN_END   = 1.15
ZOOM_OUT_BASE = 1.15   # base enlarged scale for simulated zoom out (crop expands)

def _zoom_in_clip(clip: VideoFileClip, start_scale: float, end_scale: float):
    if clip.duration <= 0:
        return clip
    def scaler(t):
        return start_scale + (end_scale - start_scale) * (t / clip.duration)
    return clip.fx(vfx.resize, scaler)

def _zoom_out_clip(clip: VideoFileClip, base_scale: float):
    """
    Simulated zoom-out: keep an enlarged version and expand a centered crop.
    Prevents black borders (never scales below 1.0).
    """
    if clip.duration <= 0:
        return clip
    big = clip.fx(vfx.resize, base_scale)
    W, H = big.w, big.h
    target_w, target_h = clip.w, clip.h  # final canvas size
    START_FACTOR = 0.90  # start slightly tighter than full frame
    start_w, start_h = int(target_w * START_FACTOR), int(target_h * START_FACTOR)

    def crop_box(t: float):
        f = t / clip.duration
        cur_w = int(start_w + (target_w - start_w) * f)
        cur_h = int(start_h + (target_h - start_h) * f)
        x1 = (W - cur_w) // 2
        y1 = (H - cur_h) // 2
        x2 = x1 + cur_w
        y2 = y1 + cur_h
        return x1, y1, x2, y2

    def make_frame(gf, t):
        x1, y1, x2, y2 = crop_box(t)
        frame = gf(t)
        return frame[y1:y2, x1:x2]

    cropped = big.fl(make_frame, apply_to=['mask']).set_duration(clip.duration)
    # Ensure expected size (may differ by 1 px due to rounding)
    if cropped.w != target_w or cropped.h != target_h:
        cropped = cropped.resize(newsize=(target_w, target_h))
    return cropped

# ---------------- Main pipeline ----------------
def build_final_video(
    job_id: str,
    upload_folder: str,
    output_folder: str,
    form: Dict,
    report: ProgressFn,
) -> str:
    files = discover_job_files(job_id, upload_folder)

    hook = files.get("hook", [None])[0]
    cta = files.get("cta", [None])[0]
    bodies = files.get("body", [])
    if not hook or not cta or not bodies:
        raise RuntimeError("Missing required clips (hook/body/cta)")

    generate_captions = form.get("generate-captions") == "on"
    blur_background = form.get("blur-background") == "on"
    normalize_audio = form.get("normalize-audio") == "on"

    logging.info(f"[{job_id}] generate_captions initial={generate_captions}")

    # Trigger keyword inputs removed; we'll match by filename later.

    music_paths = files.get("music", [])
    broll_paths = files.get("broll", [])
    overlay_paths = files.get("overlay", [])
    logging.info(f"[{job_id}] media counts: music={len(music_paths)} broll={len(broll_paths)} overlay={len(overlay_paths)}")

    # ...existing code loading music/broll/overlay...

    report(8, "load_clips")
    ordered = [hook] + bodies + [cta]
    logging.info(f"[{job_id}] ordered clip count: hook=1 bodies={len(bodies)} cta=1")
    base_clips: List[VideoFileClip] = [VideoFileClip(p) for p in ordered]

    report(15, "trim_silence")
    # Silence trimming (adjust parameters here if timing too aggressive)
    trimmed = [
        remove_silence_from_clip(
            c,
            min_silence_len_ms=SILENCE_MIN_LEN_MS_DEFAULT,
            silence_thresh_db=SILENCE_THRESH_DB_DEFAULT,
            padding_ms=SILENCE_PADDING_MS_DEFAULT,
            max_removal_ratio=SILENCE_MAX_REMOVAL_RATIO,
        )
        for c in base_clips
    ]

    # Adapt EACH trimmed clip to vertical BEFORE zoom (prevents black edgess)
    adapted = [adapt_clip_to_vertical(c, TARGET_SIZE, blur_background) for c in trimmed]

    # Apply alternating zoom only to body clips (indices 1..len(bodies))
    if len(bodies) > 0:
        for i in range(len(bodies)):
            idx = 1 + i
            if idx >= len(adapted) - 1:
                break  # skip cta
            body_clip = adapted[idx]
            if i % 2 == 0:   # zoom in
                adapted[idx] = _zoom_in_clip(body_clip, ZOOM_IN_START, ZOOM_IN_END)
            else:            # simulated zoom out
                adapted[idx] = _zoom_out_clip(body_clip, ZOOM_OUT_BASE)

    report(25, "concatenate")
    combined = concatenate_videoclips(adapted, method="compose")

    if combined.duration > 300:  # 5 minutes
        logging.warning("Trimming to 5 minute maximum.")
        combined = combined.subclip(0, 300)

    report(32, "aspect")
    combined = adapt_clip_to_vertical(combined, TARGET_SIZE, blur_background)

    # (Ensure audio fps before extraction, if not already)
    if combined.audio:
        try:
            combined = combined.set_audio(combined.audio.set_fps(44100))
        except Exception:
            pass

    temp_audio_path = os.path.join(output_folder, f"{job_id}_audio_tmp.wav")
    try:
        if combined.audio:
            combined.audio.write_audiofile(temp_audio_path, fps=44100, verbose=False, logger=None)
        else:
            logging.warning(f"[{job_id}] No audio track present before transcription.")
    except Exception as e:
        logging.warning(f"[{job_id}] Audio extract failed: {e}")

    segments: List[Dict] = []
    if generate_captions:
        report(40, "transcribe")
        segments = transcribe_audio(temp_audio_path)
        logging.info(f"[{job_id}] transcription segments={len(segments)}")
        if segments[:3]:
            logging.info(f"[{job_id}] first segments sample=" +
                         "; ".join(f"{round(s['start'],2)}-{round(s['end'],2)} '{s['text'][:30]}'"
                                   for s in segments[:3]))
        if not segments:
            generate_captions = False
            logging.info(f"[{job_id}] disabling captions (no segments)")
            report(42, "no_captions")

    caption_overlays: List[VideoFileClip] = []
    if generate_captions:
        report(48, "captions")
        caption_overlays = build_caption_clips(
            segments, (combined.w, combined.h), font_size_px=CAPTION_FONT_SIZE
        )
        logging.info(f"[{job_id}] caption clips created={len(caption_overlays)}")

    # Helper to match filename with keyword
    def find_media_for_keyword(paths: List[str], kw: str) -> Optional[str]:
        kw = kw.lower()
        for p in paths:
            if kw in os.path.basename(p).lower():
                return p
        return None

    overlay_video_elements: List[VideoFileClip] = []
    # Replace the whole matching/placement block with the following:
    if segments and (broll_paths or overlay_paths):
        report(55, "media_match")

        import re

        def extract_original_tokens(path: str):
            base = os.path.basename(path)
            parts = base.split("_", 2)
            cand = parts[2] if len(parts) == 3 else base
            name = os.path.splitext(cand)[0].lower()
            name = re.sub(r"_(\d+)$", "", name)  # strip trailing _1
            raw = re.split(r"[^a-z0-9]+", name)
            toks, seen = [], set()
            for t in raw:
                if len(t) >= BROLL_MIN_TOKEN_LEN and not t.isdigit() and t not in seen:
                    seen.add(t); toks.append(t)
            return toks

        def tokenize_words(text: str) -> List[str]:
            return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]

        def estimate_word_start(seg: Dict, matched_token: str) -> float:
            # Approximate matched word time uniformly across the segment
            words = tokenize_words(seg["text"])
            if not words:
                return seg["start"]
            try:
                idx = words.index(matched_token)
            except ValueError:
                idx = 0
            dur = max(0.01, seg["end"] - seg["start"])
            per = dur / max(1, len(words))
            return seg["start"] + idx * per

        broll_tokens = {p: extract_original_tokens(p) for p in broll_paths}
        overlay_tokens = {p: extract_original_tokens(p) for p in overlay_paths}

        if LOG_BROLL_MATCH:
            logging.info(f"[{job_id}] b-roll tokens: " + "; ".join(f"{os.path.basename(p)}={t}" for p,t in broll_tokens.items()))
            logging.info(f"[{job_id}] overlay tokens: " + "; ".join(f"{os.path.basename(p)}={t}" for p,t in overlay_tokens.items()))

        # Maintain separate lists so we can shorten the last b-roll safely
        broll_elements: List[VideoFileClip] = []
        overlay_elements: List[VideoFileClip] = []
        scheduled_brolls: List[Dict] = []  # [{start,end,clip,idx,path}]
        used_broll: set = set()
        # ADD overlay scheduling state and helper
        scheduled_overlays: List[Dict] = []  # [{start,end,clip,idx,path}]
        used_overlay: set = set()

        def schedule_overlay(clip: VideoFileClip, start_at: float, end_at: float, path: str):
            nonlocal overlay_elements, scheduled_overlays
            # Interrupt previous overlay if overlapping
            if scheduled_overlays:
                last = scheduled_overlays[-1]
                if start_at < last["end"]:
                    new_dur = max(0.05, start_at - last["start"])
                    shortened = last["clip"].set_duration(new_dur)
                    overlay_elements[last["idx"]] = shortened
                    last["clip"] = shortened
                    last["end"] = last["start"] + new_dur
            dur = max(0.05, end_at - start_at)
            clip = clip.set_start(start_at).set_duration(dur).set_position("center")
            overlay_elements.append(clip)
            scheduled_overlays.append({
                "start": start_at, "end": start_at + dur, "clip": clip,
                "idx": len(overlay_elements) - 1, "path": path
            })

        # ADD THIS: schedule_broll helper (interrupts prior b-roll and any current overlay)
        def schedule_broll(br_clip: VideoFileClip, start_at: float, end_at: float, path: str):
            nonlocal broll_elements, scheduled_brolls, overlay_elements, scheduled_overlays
            # Interrupt previous b-roll if overlapping
            if scheduled_brolls:
                last = scheduled_brolls[-1]
                if start_at < last["end"]:
                    new_dur = max(0.05, start_at - last["start"])
                    if new_dur < 0.05:
                        new_dur = 0.05
                        start_at = last["start"] + 0.05
                    shortened = last["clip"].set_duration(new_dur)
                    broll_elements[last["idx"]] = shortened
                    last["clip"] = shortened
                    last["end"] = last["start"] + new_dur
            # Also interrupt any current overlay
            if scheduled_overlays:
                o_last = scheduled_overlays[-1]
                if start_at < o_last["end"] and start_at > o_last["start"]:
                    new_dur = max(0.05, start_at - o_last["start"])
                    shortened = o_last["clip"].set_duration(new_dur)
                    overlay_elements[o_last["idx"]] = shortened
                    o_last["clip"] = shortened
                    o_last["end"] = o_last["start"] + new_dur

            dur = max(0.05, end_at - start_at)
            br_clip = br_clip.set_start(start_at).set_duration(dur).set_position("center")
            broll_elements.append(br_clip)
            scheduled_brolls.append({
                "start": start_at, "end": start_at + dur, "clip": br_clip,
                "idx": len(broll_elements) - 1, "path": path
            })

        for seg in segments:
            seg_text_raw = seg["text"]
            seg_words = tokenize_words(seg_text_raw)
            seg_word_set = set(seg_words)
            seg_dur = max(0.01, seg["end"] - seg["start"])

            # Collect all b-roll triggers in this segment, ordered by token position
            events = []  # [(token_index, path, token)]
            for p, toks in broll_tokens.items():
                if BROLL_UNIQUE and p in used_broll:
                    continue
                try:
                    idxs = [seg_words.index(t) for t in toks if t in seg_words]
                except ValueError:
                    idxs = []
                if idxs:
                    events.append((min(idxs), p, seg_words[min(idxs)]))

            # Sort by first time within the segment (earliest word first)
            events.sort(key=lambda x: x[0])

            if events:
                # Schedule each b-roll event in order; interrupt rule handled by schedule_broll()
                for _, path, tok in events:
                    try:
                        br_raw = VideoFileClip(path)
                        if AUTO_ROTATE_BROLL and getattr(br_raw, "rotation", 0) in (90, 270):
                            logging.info(f"[{job_id}] rotate b-roll {os.path.basename(path)} by {br_raw.rotation}")
                            br_raw = br_raw.rotate(br_raw.rotation)

                        token_start = estimate_word_start(seg, tok)
                        start_at = max(0.0, token_start - BROLL_TRIGGER_EARLY_SEC)

                        desired = BROLL_FIXED_DUR
                        display_dur = min(desired, float(br_raw.duration or desired))
                        end_at = min(start_at + display_dur, float(combined.duration))
                        if end_at <= start_at:
                            continue

                        # Size per mode
                        if BROLL_MODE == "cover":
                            scale = max(combined.w / br_raw.w, combined.h / br_raw.h)
                            if scale > 1 and scale > BROLL_MAX_UPSCALE: scale = BROLL_MAX_UPSCALE
                            temp = br_raw.subclip(0, end_at - start_at).resize(scale)
                            x1 = max(0, (temp.w - combined.w)//2)
                            y1 = max(0, (temp.h - combined.h)//2)
                            br  = temp.crop(x1=x1, y1=y1, x2=x1+combined.w, y2=y1+combined.h)
                        elif BROLL_MODE == "stretch":
                            br = br_raw.subclip(0, end_at - start_at).resize(newsize=(combined.w, combined.h))
                        else:
                            scale_w = combined.w / br_raw.w
                            scale_h = combined.h / br_raw.h
                            scale = min(scale_w, scale_h)
                            if scale > 1 and scale > BROLL_MAX_UPSCALE: scale = BROLL_MAX_UPSCALE
                            inner = br_raw.subclip(0, end_at - start_at).resize(scale) if scale != 1 else br_raw.subclip(0, end_at - start_at)
                            if inner.w == combined.w and inner.h == combined.h:
                                br = inner
                            else:
                                bg = ColorClip(size=(combined.w, combined.h), color=(0,0,0)).set_duration(inner.duration)
                                br = CompositeVideoClip([bg, inner.set_position("center")],
                                                        size=(combined.w, combined.h)).set_duration(inner.duration)

                        schedule_broll(br, start_at, end_at, path)
                        if BROLL_UNIQUE:
                            used_broll.add(path)
                        if LOG_BROLL_MATCH:
                            logging.info(f"[{job_id}] B-ROLL {os.path.basename(path)} tok='{tok}' @{start_at:.2f}s dur={(end_at-start_at):.2f}s")
                    except Exception as e:
                        logging.info(f"[{job_id}] b-roll insert failed: {e}")

            # NEW: schedule overlays for this segment (early start, fixed 3s, interrupt, unique, multiple per segment)
            overlay_events = []
            for p, toks in overlay_tokens.items():
                if OVERLAY_UNIQUE and p in used_overlay:
                    continue
                try:
                    idxs = [seg_words.index(t) for t in toks if t in seg_words]
                except ValueError:
                    idxs = []
                if idxs:
                    overlay_events.append((min(idxs), p, seg_words[min(idxs)]))

            overlay_events.sort(key=lambda x: x[0])

            if overlay_events:
                for _, path, tok in overlay_events:
                    try:
                        token_start = estimate_word_start(seg, tok)
                        start_at = max(0.0, token_start - OVERLAY_TRIGGER_EARLY_SEC)
                        end_at = min(start_at + OVERLAY_FIXED_DUR, float(combined.duration))
                        if end_at <= start_at:
                            continue
                        ov = rounded_image_clip(path, end_at - start_at, target_height=int(combined.h * 0.5))
                        schedule_overlay(ov, start_at, end_at, path)
                        if OVERLAY_UNIQUE:
                            used_overlay.add(path)
                        if LOG_BROLL_MATCH:
                            logging.info(f"[{job_id}] OVERLAY {os.path.basename(path)} tok='{tok}' @{start_at:.2f}s dur={(end_at-start_at):.2f}s")
                    except Exception as e:
                        logging.info(f"[{job_id}] overlay insert failed: {e}")

        # Merge into one list for final composition (b-rolls last = on top)
        overlay_video_elements = overlay_elements + broll_elements

    report(63, "compose")
    # IMPORTANT: put captions LAST so they are on top
    video_layers = [combined] + overlay_video_elements + caption_overlays
    logging.info(f"[{job_id}] layers: base + brolls({len([c for c in overlay_video_elements if hasattr(c,'size')])}) + captions({len(caption_overlays)})")
    final_video = CompositeVideoClip(video_layers, size=(combined.w, combined.h))

    report(72, "audio_mix")
    final_audio = final_video.audio
    if music_paths:
        try:
            music = AudioFileClip(music_paths[0]).volumex(0.10)
            if music.duration < final_video.duration:
                loops = math.ceil(final_video.duration / music.duration)
                from moviepy.editor import concatenate_audioclips
                music = concatenate_audioclips([music] * loops).subclip(0, final_video.duration)
            final_audio = CompositeAudioClip([final_audio, music.set_duration(final_video.duration)])
        except Exception as e:
            logging.debug(f"Music mix fail: {e}")

    if normalize_audio:
        report(78, "normalize")
        try:
            temp_vid = final_video.set_audio(final_audio)
            temp_vid = normalize_audio_level(temp_vid)
            final_audio = temp_vid.audio
        except Exception as e:
            logging.debug(f"Normalize fail: {e}")

    final_video = final_video.set_audio(final_audio)

    report(90, "encode")
    out_name = f"{job_id}_final.mp4"
    out_path = os.path.join(output_folder, out_name)
    final_video.write_videofile(
        out_path,
        codec="libx264",
        audio_codec="aac",
        threads=1,
        fps=24,
        preset="veryfast",
        verbose=False,
        logger=None,
    )

    # Cleanup
    try: os.remove(temp_audio_path)
    except: pass
    for c in base_clips:
        try: c.close()
        except: pass
    # Close overlays & captions
    for c in caption_overlays + overlay_video_elements:
        try: c.close()
        except: pass
    try: final_video.close()
    except: pass

    report(100, "done")
    return out_name