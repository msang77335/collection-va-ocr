"""
Auto rotate ID card images.

Strategy:
  1. Detect the card contour using edge detection.
  2. Extract the rotation angle from the minimum-area bounding rectangle.
  3. Apply rotation to the original image (no cropping).
  4. Ensure landscape orientation matching the template (width > height).
"""

import sys
import os
import math
import argparse
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate the image by the given angle (in degrees) around its center.
    Does NOT crop - preserves the full image with background fill.
    """
    if abs(angle) < 0.5:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Calculate new image dimensions to fit the rotated image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust the rotation matrix to account for translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))
    return rotated


def ensure_landscape(image: np.ndarray) -> np.ndarray:
    """
    Rotate 90° CW if the image is taller than wide, so the card is landscape.
    Standard ID card aspect ratio: ~1.586 (85.6 mm × 53.98 mm, CR-80).
    """
    h, w = image.shape[:2]
    if h > w:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def fix_upside_down(image: np.ndarray) -> np.ndarray:
    """
    Detect if the card text area is mostly in the bottom half (upside-down)
    by comparing average brightness in top vs bottom region.
    For typical ID card backs the logo/emblem tends to be in the top-right
    corner; the bottom strip is lighter (background).  We use a simple
    heuristic: if the upper half is brighter than the lower, rotate 180°.
    (Works well for documents on dark/colored backgrounds.)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    mid = h // 2
    top_mean = gray[:mid, :].mean()
    bot_mean = gray[mid:, :].mean()
    # Heuristic: top should be lighter (card body) for the back side.
    # If bottom is brighter, the card is flipped.
    if bot_mean > top_mean + 10:
        image = cv2.rotate(image, cv2.ROTATE_180)
    return image


# ---------------------------------------------------------------------------
# Card detection
# ---------------------------------------------------------------------------

def resolve_path(path: str) -> str:
    """
    Resolve an image path, searching common fallback locations when the
    given path does not exist:
      1. As-is
      2. Relative to the script's directory
      3. Inside a 'Data/' subdirectory next to the script
    """
    if os.path.isfile(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, path),
        os.path.join(script_dir, "Data", os.path.basename(path)),
        os.path.join("Data", os.path.basename(path)),
    ]
    for c in candidates:
        if os.path.isfile(c):
            print(f"[INFO] Resolved '{path}' → '{c}'")
            return c
    return path  # let cv2.imread produce its own error


def detect_rotation_angle(gray: np.ndarray) -> float:
    """
    Detect the rotation angle of the card by analyzing the largest contour.
    Returns the angle in degrees that should be applied to straighten the card
    to landscape orientation (width > height).
    """
    h, w = gray.shape
    img_area = h * w

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try multiple Canny thresholds to maximize detection robustness
    for lo, hi in [(50, 150), (10, 50), (80, 200), (30, 90)]:
        edges = cv2.Canny(blurred, lo, hi)
        # Dilate to close small gaps at card edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours[:10]:
            area = cv2.contourArea(cnt)
            if area < img_area * 0.05:
                break

            rect = cv2.minAreaRect(cnt)
            # rect format: (center, (w, h), angle)
            center, (box_w, box_h), angle = rect
            
            # minAreaRect angle is in [-90, 0]
            # Negate it to get the correction angle we need to apply
            # (if minAreaRect says rotate -45 to align, we need +45 to rotate back)
            correction_angle = -angle
            
            return correction_angle

    return 0.0


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def process_image(
    input_path: str,
    output_path: str,
    debug: bool = False,
) -> str:
    """
    Load *input_path*, auto-rotate the ID card, save to *output_path*.
    Returns the output path on success.
    """
    input_path = resolve_path(input_path)
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {input_path}")

    orig_h, orig_w = image.shape[:2]

    # Downscale for detection if very large
    MAX_DIM = 1200
    scale = min(MAX_DIM / orig_w, MAX_DIM / orig_h, 1.0)
    if scale < 1.0:
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = image.copy()
        scale = 1.0

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Detect rotation angle
    angle = detect_rotation_angle(gray)

    # Rotate the original full-resolution image
    result = rotate_image(image, angle)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, result)

    print(f"[OK] angle={angle:.1f}°  input={os.path.basename(input_path)}")
    print(f"     output={output_path}  size={result.shape[1]}x{result.shape[0]}")

    return output_path


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------

def _save_debug_rotation(original, small, gray, output_path):
    """Save a debug visualization of the rotation detection."""
    base, ext = os.path.splitext(output_path)
    debug_path = base + "_debug" + ext

    h, w = gray.shape
    img_area = h * w

    vis = small.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = box.astype(int)
        cv2.polylines(vis, [box], True, (0, 255, 0), 3)

    cv2.imwrite(debug_path, vis)
    print(f"     debug={debug_path}")


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def batch_process(input_dir: str, output_dir: str, debug: bool = False):
    """Process all JPEG/PNG images in *input_dir*, rotating them."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    paths = [
        os.path.join(input_dir, f)
        for f in sorted(os.listdir(input_dir))
        if os.path.splitext(f)[1].lower() in exts
    ]
    if not paths:
        print(f"No images found in {input_dir}")
        return

    for p in paths:
        name = os.path.basename(p)
        out = os.path.join(output_dir, name)
        try:
            process_image(p, out, debug=debug)
        except Exception as exc:
            print(f"[ERROR] {name}: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Auto-rotate ID card images."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input", help="Single input image path")
    group.add_argument("-d", "--dir", help="Input directory (batch mode)")

    parser.add_argument(
        "-o", "--output",
        help=(
            "Output file path (single mode) or output directory (batch mode). "
            "Defaults to <input>_out.jpg or ./Output/"
        ),
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Save a debug image showing detected card contour"
    )

    args = parser.parse_args()

    if args.input:
        resolved = resolve_path(args.input)
        base, ext = os.path.splitext(resolved)
        out = args.output or (base + "_out" + (ext or ".jpg"))
        process_image(resolved, out, debug=args.debug)
    else:
        out_dir = args.output or os.path.join(args.dir, "Output")
        batch_process(args.dir, out_dir, debug=args.debug)


if __name__ == "__main__":
    main()
