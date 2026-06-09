"""
OCR for Chinese ID card images (居民身份证).

Extracts structured fields from both sides of the card:
  Front (正面): 姓名, 性别, 民族, 出生, 住址, 公民身份号码
  Back  (背面): 签发机关, 有效期限

Dependencies:
    pip install paddlepaddle paddleocr
  OR (fallback):
    pip install easyocr

Usage:
    # Single image
    python3 ocr_card.py -i output_2.jpg

    # Batch (all images in a folder)
    python3 ocr_card.py -d ./ -o results.json

    # Raw OCR only (no field parsing)
    python3 ocr_card.py -i output_2.jpg --raw
"""

import os
import re
import sys
import json
import argparse
import tempfile
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# OCR backend loader (PaddleOCR preferred, EasyOCR fallback)
# ---------------------------------------------------------------------------

def _load_paddle():
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang="ch",
        text_detection_model_name="PP-OCRv4_mobile_det",
        text_recognition_model_name="PP-OCRv4_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )

    def _run(image_path: str):
        result = ocr.predict(image_path)
        lines = []
        if result and result[0]:
            res = result[0]
            texts  = res.get("rec_texts",  [])
            scores = res.get("rec_scores", [])
            polys  = res.get("rec_polys",  [])
            for i, text in enumerate(texts):
                conf = scores[i] if i < len(scores) else 0.0
                poly = polys[i]  if i < len(polys)  else []
                if len(poly) >= 4:
                    y_center = sum(float(pt[1]) for pt in poly) / len(poly)
                    x_center = sum(float(pt[0]) for pt in poly) / len(poly)
                else:
                    y_center = x_center = 0
                lines.append({
                    "text": text.strip(),
                    "conf": round(float(conf), 3),
                    "x": round(x_center),
                    "y": round(y_center),
                })
        return sorted(lines, key=lambda r: (round(r["y"] / 20), r["x"]))

    return _run


def _load_easyocr():
    import easyocr
    reader = easyocr.Reader(["ch_sim", "en"], gpu=False, verbose=False)

    def _run(image_path: str):
        result = reader.readtext(image_path, detail=1)
        lines = []
        for (box, text, conf) in result:
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            lines.append({
                "text": text.strip(),
                "conf": round(float(conf), 3),
                "x": round(sum(xs) / len(xs)),
                "y": round(sum(ys) / len(ys)),
            })
        return sorted(lines, key=lambda r: (round(r["y"] / 20), r["x"]))

    return _run


def get_ocr_engine():
    """Return the best available OCR runner function."""
    try:
        fn = _load_paddle()
        print("[INFO] Using PaddleOCR engine")
        return fn
    except ImportError:
        pass
    try:
        fn = _load_easyocr()
        print("[INFO] Using EasyOCR engine (fallback)")
        return fn
    except ImportError:
        pass
    print("[ERROR] No OCR engine found.")
    print("        Install one of:")
    print("          pip install paddlepaddle paddleocr")
    print("          pip install easyocr")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

# Regex patterns
_ID_NUM_RE    = re.compile(r"\b\d{17}[\dXx]\b")
_DATE_RE      = re.compile(r"\d{4}[.。\-/年]\d{1,2}[.。\-/月]\d{1,2}[日]?")
_VALIDITY_RE  = re.compile(r"\d{4}[.\-]\d{1,2}[.\-]\d{1,2}\s*[-–—~～至]\s*\d{4}[.\-]\d{1,2}[.\-]\d{1,2}|长期")
_DATE_PART_RE = re.compile(r"\d{4}[.\-]?\d{1,2}[.\-]?\d{1,2}")  # flexible: YYYY.MM.DD or YYYYMM.DD or YYYY-MM.DD
_GENDER_RE    = re.compile(r"[男女]")
_ETHNIC_RE    = re.compile(r"[族]")  # 汉族, 回族, etc.


def _closest_value(lines: list, label_y: int, label_x: int) -> str:
    """Return text of the line closest (in the same row) to the label."""
    same_row = [l for l in lines
                if abs(l["y"] - label_y) < 30 and l["x"] > label_x]
    if same_row:
        return min(same_row, key=lambda l: l["x"])["text"]
    return ""


def _lines_after_label(lines: list, label_keywords: list, max_below: int = 2) -> list:
    """Collect text lines that appear just below or beside a label keyword."""
    result = []
    for i, line in enumerate(lines):
        text = line["text"]
        if any(kw in text for kw in label_keywords):
            # Value on the same line (after the keyword)
            value_inline = re.sub("|".join(re.escape(k) for k in label_keywords), "", text).strip()
            if value_inline:
                result.append(value_inline)
            # Lines directly below
            for j in range(i + 1, min(i + 1 + max_below, len(lines))):
                nxt = lines[j]["text"]
                if any(kw in nxt for kw in ["姓名","性别","民族","出生","住址","公民","签发","有效"]):
                    break
                if nxt:
                    result.append(nxt)
            break
    return result


def _detect_side(lines: list) -> str:
    """Detect front or back of the card from the OCR text."""
    all_text = " ".join(l["text"] for l in lines)
    
    # Check for back-side keywords (with fuzzy tolerance for OCR misreads)
    # Common variants: 有效→有艾, 期限→期展, 签→岱/釜/盏, 发→弋, etc.
    has_validity = any(v in all_text for v in [
        "有效期限", "有荭期限", "有艾期展", "有艾期限"
    ]) or any("有" in l["text"] and ("期限" in l["text"] or "期展" in l["text"]) for l in lines)
    
    # Also check for heavily garbled: 有 + [1-2 chars] + 期 pattern (catches "有<期M", etc.)
    if not has_validity:
        has_validity = bool(re.search(r'有.{0,2}期', all_text))
    
    has_agency = any(a in all_text for a in [
        "签发机关", "釜岌机关", "岱发机关", "签发飙关", "釜发机关"
    ])
    
    # Also check for heavily garbled: [任何字符]发机 or [任何字符]弋机 pattern (catches "盏弋机<", etc.)
    if not has_agency:
        has_agency = bool(re.search(r'[\u4e00-\u9fff](发|弋).{0,2}机', all_text))
    
    has_id_back = "居民身份证" in all_text or "居民身份" in all_text
    
    # Special: check for "长期" or its variants (catches "长朋", "长周", etc.)
    has_longterm = any(l in all_text for l in ["长期", "长朋", "长周"]) or bool(re.search(r'长[^0-9]{0,1}$', all_text))
    
    if has_validity or has_agency or has_id_back or has_longterm:
        # Check if it also has front markers
        if "姓名" in all_text or "公民身份号码" in all_text:
            return "front"
        return "back"
    
    if "姓名" in all_text or "公民身份号码" in all_text:
        return "front"
    
    return "unknown"


def parse_front(lines: list) -> dict:
    """Extract fields from the front side."""
    all_text = " ".join(l["text"] for l in lines)
    result = {"side": "front"}

    # 姓名
    name_parts = _lines_after_label(lines, ["姓名"])
    result["姓名"] = name_parts[0] if name_parts else ""

    # 性别 & 民族 (often on the same line: "性别 男  民族 汉")
    for line in lines:
        t = line["text"]
        if "性别" in t:
            m = _GENDER_RE.search(t)
            if m:
                result["性别"] = m.group()
            # Ethnicity on same line
            ethnic_match = re.search(r"民族\s*([^\s]+族?)", t)
            if ethnic_match:
                result["民族"] = ethnic_match.group(1)
            break
    if "性别" not in result:
        result["性别"] = ""
    if "民族" not in result:
        result["民族"] = ""

    # 出生
    birth_parts = _lines_after_label(lines, ["出生"])
    birth_str = " ".join(birth_parts)
    m = _DATE_RE.search(birth_str) or _DATE_RE.search(all_text)
    result["出生"] = m.group() if m else ""

    # 住址
    addr_lines = _lines_after_label(lines, ["住址"], max_below=3)
    result["住址"] = "".join(addr_lines)

    # 公民身份号码
    m = _ID_NUM_RE.search(all_text)
    result["公民身份号码"] = m.group() if m else ""

    return result


def _normalize_date(date_str: str) -> str:
    """Normalize various date formats to YYYY.MM.DD."""
    date_str = date_str.strip()
    
    # Handle YYYYMM.DD (e.g. 201801.16 → 2018.01.16)
    m = re.match(r"(\d{4})(\d{2})\.(\d{2})", date_str)
    if m:
        return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
    
    # Handle YYYY.MMDD (e.g. 2022.0125 → 2022.01.25)
    m = re.match(r"(\d{4})\.(\d{2})(\d{2})$", date_str)
    if m:
        return f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
    
    # Handle YYYY-MM.DD (e.g. 2018-01.16 → 2018.01.16)
    m = re.match(r"(\d{4})[-/](\d{1,2})\.(\d{1,2})", date_str)
    if m:
        return f"{m.group(1)}.{m.group(2):0>2}.{m.group(3)}"
    
    # Replace all separators with dots
    normalized = re.sub(r"[-/年月]", ".", date_str)
    return normalized.rstrip("。日")


def _assemble_validity(parts: list) -> str:
    """Reassemble 有效期限 date range from possibly-fragmented OCR lines."""
    combined = " ".join(p for p in parts if p)

    # Pre-normalize: expand YYYY.MMDD → YYYY.MM.DD (e.g. 2022.0125 → 2022.01.25)
    combined = re.sub(r"(\d{4})\.(\d{2})(\d{2})", r"\1.\2.\3", combined)
    # Pre-normalize: expand YYYYMM.DD → YYYY.MM.DD (e.g. 202302.06 → 2023.02.06)
    combined = re.sub(r"(\d{4})(\d{2})\.(\d{2})", r"\1.\2.\3", combined)
    # Fix OCR year misread: 3xxx → 2xxx (OCR often reads 2 as 3, e.g. 3016 → 2016)
    combined = re.sub(r'\b3(0\d{2}|1[0-2]\d)\b', r'2\1', combined)
    # Fix commas used as decimal separator in dates (e.g. 10,19 → 10.19)
    combined = re.sub(r'(\d+),(\d+)', r'\1.\2', combined)
    # Fix colons used as date separators (e.g. "2035:07.10" → "2035.07.10")
    combined = re.sub(r'(\d+):(\d+)', r'\1.\2', combined)
    # Fix space between date numbers (e.g. "2026 10.19" → "2026.10.19")
    combined = re.sub(r'(\d{4})\s+(\d{1,2})', r'\1.\2', combined)
    combined = re.sub(r'(\d{1,2})\s+(\d{1,2})(?=\b)', r'\1.\2', combined)
    # Strip trailing single letter OCR noise after a dot (e.g. "2026.10.i" → "2026.10")
    combined = re.sub(r'\.(\s*[a-zA-Z])(?=\s|$|-)', '', combined)

    # 1. Direct match (standard case)
    m = _VALIDITY_RE.search(combined)
    if m:
        return m.group().strip()

    # 2. 长期 special case (including OCR variants like "长朋", "长周")
    has_longterm = "长期" in combined or bool(re.search(r'长[^0-9]{0,1}(?:\s|$)', combined))
    if has_longterm:
        start = _DATE_PART_RE.search(combined)
        if start:
            return f"{_normalize_date(start.group())}-长期"
        return "长期"

    # 3. Two complete date fragments (but check if end date is complete)
    dates = _DATE_PART_RE.findall(combined)
    if len(dates) >= 2:
        # Check if end date is complete (YYYY.MM.DD = 3 parts)
        end_date_str = dates[1]
        # Count numeric parts by splitting on . or -
        parts_count = len(re.split(r'[.\-]', end_date_str))
        
        if parts_count >= 3 or len(dates) > 2:  # complete, or we have extra fragments
            start = _normalize_date(dates[0])
            end = _normalize_date(dates[1])
            return f"{start}-{end}"
        # else: end date incomplete (like "2042.01"), continue to step 4

    # 4. Partial end-date: "2022.01.25-" + "2042.01" + <noise> + "25"
    start_m = re.search(r"(\d{4}[.\-]\d{1,2}[.\-]\d{1,2})\s*[-–—~～至]", combined)
    if start_m:
        after = combined[start_m.end():]
        partial = re.search(r"(\d{4}[.\-/]?\d{1,2})[^0-9]{0,8}(\d{1,2})\b", after)
        if partial:
            start_norm = _normalize_date(start_m.group(1))
            end_part = partial.group(1)
            end_day = partial.group(2)
            end_norm = _normalize_date(f"{end_part}.{end_day}")
            return f"{start_norm}-{end_norm}"

    return combined.strip()


def parse_back(lines: list) -> dict:
    """Extract fields from the back side."""
    result = {"side": "back"}

    # 签发机关 — try exact match, then fuzzy variants, then positional fallback
    _CARD_HEADER_RE = re.compile(r'居民身份证|中华人民共和国|RESIDENT|IDENTITY')
    agency_parts = _lines_after_label(lines, ["签发机关", "釜岌机关", "签发飙关", "岱发机关", "釜发机关"])
    # Filter out card header/title noise that may follow the label when card is rotated
    # Also strip any date-formatted lines (date belongs to 有效期限, not 签发机关)
    _DATE_LINE_RE = re.compile(r'\d{4}[.\-]\d{1,2}[.\-]\d{1,2}')
    agency_parts = [p for p in agency_parts
                    if not _CARD_HEADER_RE.search(p) and not _DATE_LINE_RE.search(p)]
    agency_text = "".join(agency_parts)
    # Validate: must contain at least one Chinese character (not just noise/digits)
    agency_valid = bool(re.search(r'[\u4e00-\u9fff]', agency_text)) and len(agency_text) >= 2
    if not agency_valid:
        # Positional fallback: look BEFORE 有效期限 label and also AFTER 签发机关 label
        # First try lines immediately after the 签发机关 label going upward (before in y order)
        for i, line in enumerate(lines):
            if any(kw in line["text"] for kw in ["签发机关", "釜岌机关", "签发飙关", "岱发机关", "釜发机关"]):
                for j in range(i - 1, max(i - 10, -1), -1):
                    candidate = lines[j]["text"].strip()
                    if (len(candidate) >= 3 and
                            re.search(r'[\u4e00-\u9fff]', candidate) and
                            not _CARD_HEADER_RE.search(candidate) and
                            not any(kw in candidate for kw in
                                    ["有效", "有荭", "签发", "@"])):
                        agency_parts = [candidate]
                        agency_valid = True
                        break
                break
    if not agency_valid:
        for i, line in enumerate(lines):
            if ("\u6709\u6548\u671f\u9650" in line["text"] or "\u6709\u836d\u671f\u9650" in line["text"]) and i > 0:
                # Search up to 20 lines before the label; require Chinese chars, skip noise
                for j in range(i - 1, max(i - 20, -1), -1):
                    candidate = lines[j]["text"].strip()
                    if (len(candidate) >= 4 and
                            re.search(r'[\u4e00-\u9fff]', candidate) and
                            not _CARD_HEADER_RE.search(candidate) and
                            not any(kw in candidate for kw in
                                    ["\u4e2d\u534e", "\u5c45\u6c11", "\u8eab\u4efd\u8bc1", "\u6709\u6548", "\u6709\u836d", "\u7b7e\u53d1", "@"])):
                        agency_parts = [candidate]
                        break
                break
    # Clean up leading/trailing connectors and non-Chinese chars from agency name
    agency_text = "".join(agency_parts)
    agency_text = re.sub(r'^[-·一•\s]+', '', agency_text)  # strip leading connectors
    agency_text = re.sub(r'[-·一•\s]+$', '', agency_text)  # strip trailing connectors
    # Strip all non-Chinese characters (keep only 中文 and digits)
    agency_text = re.sub(r'[A-Za-z0-9]', '', agency_text)
    agency_text = re.sub(r'\s+', '', agency_text)  # remove all spaces

    # Last resort: scan ALL lines for a bureau name pattern (handles garbled labels)
    if not agency_text:
        # Match: 2-12 Chinese chars ending with 公安局/分局/派出所/etc.
        # Use a tighter prefix: require at least one real city/district char before the suffix
        _BUREAU_RE2 = re.compile(r'[\u4e00-\u9fff]{2,10}(?:市|县|区|省|自治区|旗)[\u4e00-\u9fff]{1,8}(?:公安局|公安分局|派出所|公安处|警察局)')
        for line in lines:
            bm = _BUREAU_RE2.search(line["text"])
            if bm and not _CARD_HEADER_RE.search(bm.group()):
                agency_text = bm.group()
                break
        # Fallback: any bureau suffix match
        if not agency_text:
            _BUREAU_RE3 = re.compile(r'[\u4e00-\u9fff]{2,12}(?:公安局|公安分局|派出所|公安处|警察局)')
            for line in lines:
                bm = _BUREAU_RE3.search(line["text"])
                if bm and not _CARD_HEADER_RE.search(bm.group()):
                    # Strip any leading garbled label chars (non-city keywords)
                    text = bm.group()
                    # Remove leading chars that look like label noise (≤2 chars before a city marker)
                    text = re.sub(r'^[\u4e00-\u9fff]{1,3}(?=[\u4e00-\u9fff]{2,}(?:市|县|区|省))', '', text)
                    agency_text = text
                    break

    result["签发机关"] = agency_text

    # 有效期限 — try exact match and fuzzy variants, collect up to 8 lines
    validity_parts = _lines_after_label(lines, ["有效期限", "有荭期限", "有艾期展", "有艾期限"], max_below=8)

    # If 签发机关 is still empty, try to find a bureau name embedded in validity lines
    if not result["签发机关"]:
        _BUREAU_RE = re.compile(r'[\u4e00-\u9fff]{2,10}(?:公安局|公安分局|派出所|公安处|警察局)')
        validity_combined = " ".join(validity_parts)
        bm = _BUREAU_RE.search(validity_combined)
        if bm:
            result["签发机关"] = bm.group()
            # Strip the agency text from validity_parts so it doesn't pollute the date
            validity_parts = [p for p in validity_parts if bm.group() not in p]

    assembled = _assemble_validity(validity_parts)

    # Fallback: date may appear BEFORE the label (e.g. after 270° rotation)
    # Triggered when assembled contains no recognisable date fragment
    if not _DATE_PART_RE.search(assembled):
        high_conf = [l["text"] for l in lines if l.get("conf", 0) > 0.2]
        assembled = _assemble_validity(high_conf)

    # Clean up: if we have "YYYY.MM.DD-长[garbage]" or "YYYY.MM.DD-长", replace with just "长期"
    if re.search(r'\d{4}\.\d{1,2}\.\d{1,2}-长(?:[^期]|$)', assembled):
        assembled = "长期"
    
    # Clean up: remove spaces/extra dots in date strings (e.g. "2014. .11.21" → "2014.11.21")
    assembled = re.sub(r'(\d{4})\.\s+\.', r'\1.', assembled)  # remove " ." pattern
    assembled = re.sub(r'(\d{4})\.\s+', r'\1.', assembled)  # remove " " after first dot
    assembled = re.sub(r'(\d{1,2})\.\s+', r'\1.', assembled)  # remove " " after month dot

    result["有效期限"] = assembled

    return result


def parse_fields(lines: list) -> dict:
    side = _detect_side(lines)
    if side == "front":
        return parse_front(lines)
    if side == "back":
        return parse_back(lines)
    # Unknown: return all text
    return {"side": "unknown", "text": [l["text"] for l in lines]}


# ---------------------------------------------------------------------------
# Multi-rotation OCR
# ---------------------------------------------------------------------------

# Keywords that indicate a valid card read (front or back)
_FRONT_KEYWORDS = ["姓名", "性别", "民族", "出生", "住址", "公民身份号码"]
_BACK_KEYWORDS  = ["有效期限", "签发机关", "居民身份证"]
_ALL_KEYWORDS   = _FRONT_KEYWORDS + _BACK_KEYWORDS

# Partial/fuzzy substrings — OCR sometimes misreads a character
_FUZZY_KEYWORDS = ["居民", "身份", "有效", "签发", "姓名", "性别", "住址", "出生"]


def _rotation_code(degrees: int):
    """Map degrees → cv2 rotation code (None = no rotation)."""
    return {
        0:   None,
        90:  cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }[degrees]


def _score_lines(lines: list) -> float:
    """
    Score OCR output by how many card keywords/substrings are present.
    Exact keywords score 1.0 each; fuzzy partial matches score 0.5 each.
    Also rewards high average confidence.
    Bonus +3.0 when a valid date range (有效期限) is found — prevents
    tilted-card cases where many labels are detected but no date is parseable.
    """
    all_text = " ".join(l["text"] for l in lines)
    score = 0.0
    score += sum(1.0 for kw in _ALL_KEYWORDS  if kw in all_text)
    score += sum(0.5 for kw in _FUZZY_KEYWORDS if kw in all_text)
    if lines:
        avg_conf = sum(l["conf"] for l in lines) / len(lines)
        score += avg_conf * 0.5   # slight bonus for high-confidence reads
    # Strong bonus when an actual parseable date range is present
    if _VALIDITY_RE.search(all_text):
        score += 3.0
    # Extra bonus for a *complete* date range (2-digit day on both ends, e.g. YYYY.MM.DD-YYYY.MM.DD)
    # Prevents truncated OCR reads (e.g. "2040.06.0") from winning over complete reads ("2040.06.01")
    _VALIDITY_COMPLETE_RE = re.compile(
        r"\d{4}[.\-]\d{1,2}[.\-]\d{2}\s*[-–—~～至]\s*\d{4}[.\-]\d{1,2}[.\-]\d{2}|长期"
    )
    if _VALIDITY_COMPLETE_RE.search(all_text):
        score += 0.5
    return score


_MAX_OCR_SIDE = 1800  # downscale large images before OCR to save memory


def _resize_for_ocr(img: np.ndarray) -> np.ndarray:
    """Downscale image so its longest side is at most _MAX_OCR_SIDE."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= _MAX_OCR_SIDE:
        return img
    scale = _MAX_OCR_SIDE / longest
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _deskew_card(img: np.ndarray) -> np.ndarray:
    """
    Detect card tilt angle via minAreaRect on the largest card-like contour
    and rotate the image straight.  Returns corrected image (or original if
    no clear card found).
    Uses multiple masks (cyan color, bright-low-sat, edge-based) for robustness.
    """
    h_img, w_img = img.shape[:2]
    scale = min(1.0, 800 / max(h_img, w_img))
    small = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1.0 else img.copy()
    sh, sw = small.shape[:2]

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    # Mask 1: cyan/blue hue (ID card background)
    mask_cyan = cv2.inRange(hsv,
                            np.array([80,  8, 120], dtype=np.uint8),
                            np.array([135, 180, 255], dtype=np.uint8))
    mask_cyan = cv2.morphologyEx(mask_cyan, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask_cyan = cv2.morphologyEx(mask_cyan, cv2.MORPH_OPEN,  kernel, iterations=2)

    # Mask 2: bright & low-saturation (white card areas vs pure-white bg)
    mask_card = cv2.bitwise_and(
        cv2.inRange(v_ch, 160, 245),  # bright but not pure 255
        cv2.inRange(s_ch, 3, 80),     # slight saturation (not pure gray/white)
    )
    mask_card = cv2.morphologyEx(mask_card, cv2.MORPH_CLOSE, kernel, iterations=4)
    mask_card = cv2.morphologyEx(mask_card, cv2.MORPH_OPEN,  kernel, iterations=2)

    # Mask 3: any non-white area (catches dark text on card)
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_small, (5, 5), 0)
    mask_dark = cv2.inRange(blurred, 0, 220)
    mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_CLOSE, kernel, iterations=5)
    mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_OPEN,  kernel, iterations=2)

    # Try masks in order of reliability
    best_mask = None
    for mask in (mask_cyan, cv2.bitwise_or(mask_cyan, mask_card), mask_dark):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area > sh * sw * 0.04:
                best_mask = mask
                break

    contours = []
    if best_mask is not None:
        contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    h_img, w_img = img.shape[:2]
    img_area = h_img * w_img

    # Pick largest contour that looks like a card (area 5-90% of image)
    best_cnt = None
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.05 or area > img_area * 0.92:
            continue
        rect = cv2.minAreaRect(cnt)
        _, (rw, rh), _ = rect
        if rw < 10 or rh < 10:
            continue
        aspect = max(rw, rh) / (min(rw, rh) + 1e-5)
        # ID card aspect ~1.58; accept 1.1 – 2.5
        if 1.1 <= aspect <= 2.5:
            best_cnt = cnt
            break

    if best_cnt is None:
        return img

    rect = cv2.minAreaRect(best_cnt)
    _, (rw, rh), angle = rect

    # minAreaRect angle convention: rotate the longer side to horizontal
    if rw < rh:
        angle = angle + 90  # portrait box → compensate
    # angle is now the tilt of the card's long axis relative to horizontal
    # Ignore if nearly aligned (< 2°) or very large (> 45° → let 4-rotation handle it)
    if abs(angle) < 2 or abs(angle) > 45:
        return img

    # Rotate the full image to straighten the card
    cx, cy = w_img / 2, h_img / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    # Expand canvas so corners are not clipped
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h_img * sin_a + w_img * cos_a)
    new_h = int(h_img * cos_a + w_img * sin_a)
    M[0, 2] += (new_w - w_img) / 2
    M[1, 2] += (new_h - h_img) / 2
    straightened = cv2.warpAffine(img, M, (new_w, new_h),
                                  borderMode=cv2.BORDER_REPLICATE)
    print(f"  [DESKEW] corrected {angle:.1f}°  ({w_img}x{h_img} → {new_w}x{new_h})")
    return straightened


def ocr_all_rotations(image_path: str, ocr_fn) -> tuple:
    """
    OCR the image in all 4 rotations (0°, 90°, 180°, 270°).
    Returns (best_lines, best_degrees) — the rotation with the most card keywords.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    img = _resize_for_ocr(img)
    img = _deskew_card(img)  # straighten tilted card before rotation search

    best_lines   = []
    best_score   = -1
    best_degrees = 0

    for degrees in (0, 90, 180, 270):
        rot_code = _rotation_code(degrees)
        rotated  = cv2.rotate(img, rot_code) if rot_code is not None else img

        # Write to a temp file so the OCR engine can read it
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cv2.imwrite(tmp_path, rotated)
            lines = ocr_fn(tmp_path)
        finally:
            os.unlink(tmp_path)

        score = _score_lines(lines)
        print(f"  [ROT {degrees:3d}°] score={score}  "
              f"text={[l['text'] for l in lines[:4]]}")

        if score > best_score:
            best_score   = score
            best_lines   = lines
            best_degrees = degrees

    print(f"  [BEST] {best_degrees}° (score={best_score})")
    return best_lines, best_degrees


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def ocr_image(image_path: str, ocr_fn, raw: bool = False,
              auto_rotate: bool = True) -> dict:
    """
    Run OCR on *image_path* and return a structured result dict.
    When *auto_rotate* is True (default), tries all 4 rotations and picks
    the one with the most card keywords.
    If *raw* is True, skip field parsing and return only raw OCR lines.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if auto_rotate:
        lines, degrees = ocr_all_rotations(image_path, ocr_fn)
    else:
        lines   = ocr_fn(image_path)
        degrees = 0

    if raw:
        return {
            "file":    os.path.basename(image_path),
            "degrees": degrees,
            "lines":   [{"text": l["text"], "conf": l["conf"]} for l in lines],
        }

    fields = parse_fields(lines)
    fields["file"]    = os.path.basename(image_path)
    fields["degrees"] = degrees
    return fields


def batch_ocr(input_dir: str, ocr_fn, raw: bool = False,
              auto_rotate: bool = True) -> list:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    paths = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in exts
    )
    if not paths:
        print(f"[WARN] No images found in {input_dir}")
        return []

    results = []
    for p in paths:
        try:
            r = ocr_image(p, ocr_fn, raw=raw, auto_rotate=auto_rotate)
            print(f"[OK] {os.path.basename(p)}")
            results.append(r)
        except Exception as e:
            print(f"[ERROR] {os.path.basename(p)}: {e}")
            results.append({"file": os.path.basename(p), "error": str(e)})
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OCR Chinese ID card images and extract structured fields."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input",  help="Single input image")
    group.add_argument("-d", "--dir",    help="Input directory (batch mode)")

    parser.add_argument("-o", "--output",
                        help="Output JSON file (default: print to stdout)")
    parser.add_argument("--raw", action="store_true",
                        help="Return raw OCR lines without field parsing")
    parser.add_argument("--no-rotate", action="store_true",
                        help="Skip multi-rotation, OCR the image as-is")
    parser.add_argument("--engine", choices=["paddle", "easyocr"],
                        help="Force a specific OCR engine")

    args = parser.parse_args()

    # Load engine
    if args.engine == "paddle":
        ocr_fn = _load_paddle()
    elif args.engine == "easyocr":
        ocr_fn = _load_easyocr()
    else:
        ocr_fn = get_ocr_engine()

    auto_rotate = not args.no_rotate

    # Process
    if args.input:
        result = ocr_image(args.input, ocr_fn, raw=args.raw,
                           auto_rotate=auto_rotate)
        output = result
    else:
        output = batch_ocr(args.dir, ocr_fn, raw=args.raw,
                           auto_rotate=auto_rotate)

    # Output
    json_str = json.dumps(output, ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"[INFO] Results saved to {args.output}")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
