import cv2
import numpy as np
from pathlib import Path


IMAGES_DIR = Path(__file__).parent / "images"
OUTPUT_DIR = Path(__file__).parent / "output"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

HOUGH_DP = 1.0
HOUGH_MIN_DIST = 25
HOUGH_PARAM1 = 100  # upper Canny threshold
HOUGH_PARAM2 = 10   # accumulator threshold; lower = more sensitive
HOUGH_MIN_RADIUS = 6
HOUGH_MAX_RADIUS = 16
HOUGH_MAX_SURROUNDING_BRIGHTNESS = 70  # mean of outer annulus; reject if brighter (off-panel)
HOUGH_MIN_CENTER_BRIGHTNESS = 110      # mean of inner disc; reject if dimmer (dust/glare, not a puck)
HOUGH_MIN_CENTER_PEAK = 160            # brightest pixel in inner disc; reject dirt that never reaches true white
HOUGH_MIN_CONTRAST = 60                # center mean - surrounding mean; allows bright pucks on over-exposed panels

# Ellipse-based puck detection (catches perspective-distorted pucks Hough misses)
ELLIPSE_THRESHOLD = 130        # bright pixels only; pucks are near-white
ELLIPSE_MIN_AREA = 20
ELLIPSE_MAX_AREA = 1500
ELLIPSE_MIN_AXIS = 4           # min minor-axis radius of fitted ellipse; rejects tiny dust blobs
ELLIPSE_MAX_AXIS = 25          # max major-axis radius of fitted ellipse; rejects large bright objects
ELLIPSE_MAX_ASPECT = 3.0       # reject elongated blobs (stickers, edge artifacts)
ELLIPSE_MIN_FILL = 0.50        # contour_area / fitted_ellipse_area; reject irregular blobs

# Sticker detection (contour-based) for sticker-vs-puck disambiguation
STICKER_THRESHOLD = 160
STICKER_CLOSE_KERNEL = 9   # bridges barcode/QR stripes so a sticker stays one blob
STICKER_MIN_AREA = 120
STICKER_MAX_AREA = 4000
STICKER_MIN_RECT_FIT = 0.80    # contour_area / minAreaRect_area; rectangle ≈ 1.0, circle ≈ 0.785
STICKER_MIN_ASPECT_RATIO = 1.3  # longer side / shorter side of fitted rect; rejects square-ish blobs


def detect_stickers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, STICKER_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (STICKER_CLOSE_KERNEL, STICKER_CLOSE_KERNEL)
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stickers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < STICKER_MIN_AREA or area > STICKER_MAX_AREA:
            continue

        # Fit an oriented rectangle (handles rotation and rounded corners).
        (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
        if rw == 0 or rh == 0:
            continue
        rect_area = rw * rh
        rect_fit = area / rect_area
        if rect_fit < STICKER_MIN_RECT_FIT:
            continue

        long_side, short_side = max(rw, rh), min(rw, rh)
        aspect_ratio = long_side / short_side
        if aspect_ratio < STICKER_MIN_ASPECT_RATIO:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        stickers.append((x, y, w, h))
    return stickers


def _point_in_bbox(px, py, bbox):
    x, y, w, h = bbox
    return x <= px <= x + w and y <= py <= y + h


def detect_pucks(image, stickers):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS,
        maxRadius=HOUGH_MAX_RADIUS,
    )

    if circles is None:
        return []

    results = []
    for cx, cy, cr in circles[0]:
        cx, cy, cr = int(cx), int(cy), int(cr)

        # Inner disc must be bright (a real puck is white, not dust/glare),
        # and must contain at least one near-white pixel (dirt never reaches true white).
        center_mask = np.zeros_like(gray)
        cv2.circle(center_mask, (cx, cy), max(int(cr * 0.5), 1), 255, -1)
        center_brightness = cv2.mean(gray, mask=center_mask)[0]
        if center_brightness < HOUGH_MIN_CENTER_BRIGHTNESS:
            continue
        _, max_val, _, _ = cv2.minMaxLoc(gray, mask=center_mask)
        if max_val < HOUGH_MIN_CENTER_PEAK:
            continue

        # Surrounding annulus must be dark in absolute terms OR sufficiently darker than the
        # center (the latter saves pucks on over-exposed panels where the "dark" panel is brighter).
        ring_mask = np.zeros_like(gray)
        cv2.circle(ring_mask, (cx, cy), int(cr * 2.0), 255, -1)
        cv2.circle(ring_mask, (cx, cy), int(cr * 1.3), 0, -1)
        if cv2.countNonZero(ring_mask) == 0:
            continue
        surrounding_brightness = cv2.mean(gray, mask=ring_mask)[0]
        if (surrounding_brightness > HOUGH_MAX_SURROUNDING_BRIGHTNESS
                and center_brightness - surrounding_brightness < HOUGH_MIN_CONTRAST):
            continue

        # Reject if the circle center sits inside a detected sticker bbox.
        if any(_point_in_bbox(cx, cy, s) for s in stickers):
            continue

        results.append((cx, cy, cr))

    # Ellipse-based pass for perspective-distorted pucks Hough won't catch.
    results.extend(_detect_pucks_ellipse(gray, results, stickers))
    return results


def _detect_pucks_ellipse(gray, existing_pucks, stickers):
    _, binary = cv2.threshold(gray, ELLIPSE_THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    extras = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < ELLIPSE_MIN_AREA or area > ELLIPSE_MAX_AREA:
            continue
        if len(contour) < 5:  # fitEllipse needs at least 5 points
            continue

        (ex, ey), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        major, minor = max(axis_a, axis_b) / 2, min(axis_a, axis_b) / 2
        if minor < ELLIPSE_MIN_AXIS or major > ELLIPSE_MAX_AXIS:
            continue
        if major / minor > ELLIPSE_MAX_ASPECT:
            continue
        if area / (np.pi * major * minor) < ELLIPSE_MIN_FILL:
            continue

        cx, cy, cr = int(ex), int(ey), int(major)

        # Skip if Hough already found this puck (within minDist).
        if any((cx - hx) ** 2 + (cy - hy) ** 2 < HOUGH_MIN_DIST ** 2
               for hx, hy, _ in existing_pucks):
            continue

        # Skip if inside a sticker bbox.
        if any(_point_in_bbox(cx, cy, s) for s in stickers):
            continue

        # Surrounding must be dark (same constraint as Hough path).
        ring_mask = np.zeros_like(gray)
        cv2.circle(ring_mask, (cx, cy), int(cr * 2.0), 255, -1)
        cv2.circle(ring_mask, (cx, cy), int(cr * 1.3), 0, -1)
        if cv2.countNonZero(ring_mask) == 0:
            continue
        if cv2.mean(gray, mask=ring_mask)[0] > HOUGH_MAX_SURROUNDING_BRIGHTNESS:
            continue

        extras.append((cx, cy, cr))
    return extras


def draw_detections(image, pucks, stickers):
    output = image.copy()
    for x, y, w, h in stickers:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(output, "sticker", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for i, (x, y, r) in enumerate(pucks, start=1):
        cv2.circle(output, (x, y), r, (0, 0, 255), 2)
        cv2.circle(output, (x, y), 1, (0, 0, 255), 2)
        cv2.putText(output, f"#{i}", (x + r + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return output


def process_image(image_path, output_dir):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  skipped (could not read)")
        return

    stickers = detect_stickers(image)
    pucks = detect_pucks(image, stickers)
    output = draw_detections(image, pucks, stickers)

    output_path = output_dir / f"{image_path.stem}.jpg"
    cv2.imwrite(str(output_path), output)
    print(f"  {len(pucks)} pucks, {len(stickers)} stickers → {output_path.name}")


def process_directory(input_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    for image_path in image_paths:
        print(f"Processing {input_dir.name}/{image_path.name}")
        process_image(image_path, output_dir)


def main():
    subdirs = sorted(p for p in IMAGES_DIR.iterdir() if p.is_dir())
    if not subdirs:
        print(f"No subdirectories found in {IMAGES_DIR}")
        return
    for subdir in subdirs:
        process_directory(subdir, OUTPUT_DIR / subdir.name)


if __name__ == "__main__":
    main()
