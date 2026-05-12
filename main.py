"""Puck / sticker detector for DAF truck underbody panels.

Pipeline (each image runs through these stages in order):

    STAGE 1 — Panel mask
        Find the large dark panel where pucks can legitimately appear.
        Everything outside this mask (side frame, scaffolding, chassis
        cutouts) is ignored in later stages. Built with simple grayscale
        + HSV thresholding followed by morphological cleanup.

    STAGE 2 — Sticker detection
        Bright rectangular labels (barcodes / QR stickers) on the panel.
        Found by binary thresholding the grayscale image, then keeping
        contours whose fitted rectangle is rectangular and elongated.
        Recorded as bounding boxes so STAGE 3 can ignore them.

    STAGE 3 — Puck detection
        Two passes that complement each other:
          a) Hough Circle Transform (cv2.HoughCircles)  — fast, finds
             well-formed round pucks viewed near head-on.
          b) Ellipse fitting on bright contours (cv2.fitEllipse) —
             catches pucks distorted by perspective that Hough misses.
        Both passes reject any candidate that:
          - sits outside the panel mask (STAGE 1)
          - sits inside a sticker bbox (STAGE 2, the "dedupe" step)
          - fails brightness/contrast sanity checks

    STAGE 4 — Visualization
        Tint the panel mask, draw sticker boxes, draw puck circles.

No machine-learning models are used — only classical OpenCV operations.
"""

import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

IMAGES_DIR = Path(__file__).parent / "images"
OUTPUT_DIR = Path(__file__).parent / "output"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------------------
# STAGE 1 constants — Panel mask
# ---------------------------------------------------------------------------
# A pixel is "panel" if it is both dark (low grayscale value) and desaturated
# (low HSV saturation — i.e. close to gray, not colorful).
PANEL_BRIGHTNESS_MAX = 75   # pixel is "panel" if grayscale value < this
PANEL_SATURATION_MAX = 110  # AND HSV saturation < this (panel is desaturated)
PANEL_CLOSE_KERNEL = 71     # morphological close: fills holes (pucks, seams) up to this size
PANEL_OPEN_KERNEL = 35      # morphological open: breaks narrow corridors connecting panel to corners
PANEL_ERODE = 6             # final shrink to stay clear of the frame transition
PANEL_MIN_AREA = 50000      # ignore tiny dark blobs; the real panel is huge
PANEL_DEBUG_OVERLAY = True  # tint the mask onto the output image for visual tuning

# Frame exclusion: the truck side frame is a bright/saturated strip that
# touches the left or right image edge. We build a mask of it and subtract
# it from the panel so dark "shoulders" near the corners don't sneak in.
FRAME_BRIGHTNESS_MIN = 110  # pixel is "frame-like" if brighter than this
FRAME_SATURATION_MIN = 140  # OR more saturated than this (blue/red/green frame paint)
FRAME_ERODE = 11            # erase isolated bright spots (pucks) before merging the frame
FRAME_CLOSE_KERNEL = 41     # merge surviving frame fragments into one component
FRAME_DILATE = 25           # how far the final frame exclusion reaches into the corners


# ---------------------------------------------------------------------------
# STAGE 2 constants — Sticker detection
# ---------------------------------------------------------------------------
# Stickers are bright, rectangular, and clearly longer than they are wide.
STICKER_THRESHOLD = 160         # binary threshold: only pixels brighter than this survive
STICKER_CLOSE_KERNEL = 9        # bridge barcode stripes so a sticker stays one blob
STICKER_MIN_AREA = 120
STICKER_MAX_AREA = 4000
STICKER_MIN_RECT_FIT = 0.80     # contour_area / minAreaRect_area; rectangle ≈ 1.0, circle ≈ 0.785
STICKER_MIN_ASPECT_RATIO = 1.3  # long side / short side; rejects square-ish blobs


# ---------------------------------------------------------------------------
# STAGE 3 constants — Puck detection
# ---------------------------------------------------------------------------
# Pass A: Hough Circle Transform parameters.
HOUGH_DP = 1.0
HOUGH_MIN_DIST = 25
HOUGH_PARAM1 = 100  # upper Canny edge threshold used internally by HoughCircles
HOUGH_PARAM2 = 10   # accumulator threshold; lower = more sensitive (more circles)
HOUGH_MIN_RADIUS = 6
HOUGH_MAX_RADIUS = 16
HOUGH_MAX_SURROUNDING_BRIGHTNESS = 70  # mean of ring around the circle; reject if bright (off-panel)
HOUGH_MIN_CENTER_BRIGHTNESS = 110      # mean inside the circle; reject if dim (dust/glare, not a puck)
HOUGH_MIN_CENTER_PEAK = 160            # brightest pixel inside; dirt never reaches near-white
HOUGH_MIN_CONTRAST = 60                # center mean - ring mean; saves pucks on over-exposed panels

# Pass B: Ellipse fitting fallback (catches perspective-distorted pucks).
ELLIPSE_THRESHOLD = 130        # bright pixels only; pucks are near-white
ELLIPSE_MIN_AREA = 20
ELLIPSE_MAX_AREA = 1500
ELLIPSE_MIN_AXIS = 4           # min minor-axis radius of fitted ellipse; rejects tiny dust blobs
ELLIPSE_MAX_AXIS = 25          # max major-axis radius of fitted ellipse; rejects large bright objects
ELLIPSE_MAX_ASPECT = 3.0       # reject elongated blobs (stickers, edge artifacts)
ELLIPSE_MIN_FILL = 0.50        # contour_area / fitted_ellipse_area; rejects irregular blobs


# ===========================================================================
# STAGE 1 — Panel mask
# ===========================================================================
# We want a binary image that is white where the dark puck panel is, and
# black everywhere else. Subsequent stages will only trust detections that
# fall inside this mask.

def _gray_world_balance(image):
    """Neutralize a per-image color cast so saturation-based thresholds behave.

    A warm/cool tint pushes HSV saturation high even on the dark panel, which
    causes the panel test (sat < PANEL_SATURATION_MAX) to reject huge regions
    and the frame test (sat > FRAME_SATURATION_MIN) to swallow them. Pulling
    the B/G/R channel means together (the "gray-world" assumption) removes
    the cast. Already-neutral images get multipliers near 1.0 and are
    effectively untouched.
    """
    b, g, r = cv2.split(image.astype(np.float32))
    mean = (b.mean() + g.mean() + r.mean()) / 3.0
    b *= mean / max(b.mean(), 1e-6)
    g *= mean / max(g.mean(), 1e-6)
    r *= mean / max(r.mean(), 1e-6)
    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)


def build_panel_mask(image):
    """Binary mask of the dark puck panel, with bright puck holes filled in.

    Anything outside this mask (side frame, scaffolding, chassis cutouts) is
    not a valid place for a puck.
    """
    # Color-balance first so saturation thresholds work on tinted images too.
    balanced = _gray_world_balance(image)
    gray = cv2.cvtColor(balanced, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(balanced, cv2.COLOR_BGR2HSV)[:, :, 1]

    # Initial guess: dark AND desaturated pixels.
    panel = ((gray < PANEL_BRIGHTNESS_MAX) & (saturation < PANEL_SATURATION_MAX))
    panel = panel.astype(np.uint8) * 255

    # CLOSE = dilate then erode. Fills small holes (pucks, stickers, weld
    # seams) so the panel becomes one solid region.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (PANEL_CLOSE_KERNEL, PANEL_CLOSE_KERNEL)
    )
    panel = cv2.morphologyEx(panel, cv2.MORPH_CLOSE, kernel)

    # OPEN = erode then dilate. Breaks thin bridges so dark corner shoulders
    # that happen to be connected to the panel via a narrow path get pinched
    # off into their own (small) component and dropped in the next step.
    if PANEL_OPEN_KERNEL > 0:
        open_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (PANEL_OPEN_KERNEL, PANEL_OPEN_KERNEL)
        )
        panel = cv2.morphologyEx(panel, cv2.MORPH_OPEN, open_k)

    # Keep only the largest connected blob — that's the panel itself.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(panel, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(panel)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < PANEL_MIN_AREA:
        return np.zeros_like(panel)
    mask = (labels == largest).astype(np.uint8) * 255

    # Fill any remaining internal holes (bright seams the close kernel
    # couldn't span). RETR_EXTERNAL ignores inner contours; redrawing each
    # outer contour as a filled shape leaves no holes inside.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    # Subtract the (dilated) side-frame mask so the bright frame and its
    # dark "shoulders" near the image corners are never considered panel.
    frame = _build_frame_mask(gray, saturation)
    filled = cv2.bitwise_and(filled, cv2.bitwise_not(frame))

    # Light final erosion: stay a few pixels back from the frame transition.
    if PANEL_ERODE > 0:
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (PANEL_ERODE * 2 + 1, PANEL_ERODE * 2 + 1)
        )
        filled = cv2.erode(filled, erode_kernel)
    return filled


def _build_frame_mask(gray, saturation):
    """Mask of the truck side frame (and scaffolding seen through cutouts).

    The frame is identified as bright-or-saturated material that touches the
    left or right image border. We then dilate it so the exclusion reaches
    into the corner shoulders.
    """
    # Bright OR colorful pixels are candidates for "frame".
    frame = ((gray > FRAME_BRIGHTNESS_MIN) | (saturation > FRAME_SATURATION_MIN))
    frame = frame.astype(np.uint8) * 255

    # Erode away isolated bright spots (the pucks themselves). A real frame
    # is thicker than a puck, so it survives erosion; pucks vanish.
    if FRAME_ERODE > 0:
        erode_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (FRAME_ERODE * 2 + 1, FRAME_ERODE * 2 + 1)
        )
        frame = cv2.erode(frame, erode_k)

    # Close to glue surviving fragments into a single side-rail component.
    close_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (FRAME_CLOSE_KERNEL, FRAME_CLOSE_KERNEL)
    )
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, close_k)

    # Keep only components that touch the left or right edge. The central
    # weld seam touches top/bottom but never the sides, so it is excluded.
    num_labels, labels = cv2.connectedComponents(frame)
    edge_labels = set(np.unique(labels[:, 0]).tolist())
    edge_labels.update(np.unique(labels[:, -1]).tolist())
    edge_labels.discard(0)
    if not edge_labels:
        return np.zeros_like(frame)
    side_frame = np.isin(labels, list(edge_labels)).astype(np.uint8) * 255

    # Dilate to extend the exclusion into the corner transition zone.
    dilate_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (FRAME_DILATE * 2 + 1, FRAME_DILATE * 2 + 1)
    )
    return cv2.dilate(side_frame, dilate_k)


def _in_panel(panel_mask, cx, cy):
    """True iff pixel (cx, cy) is inside the panel mask."""
    h, w = panel_mask.shape
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return False
    return panel_mask[cy, cx] != 0


# ===========================================================================
# STAGE 2 — Sticker detection
# ===========================================================================
# Stickers are bright (white background), rectangular, and longer than they
# are wide. We threshold the grayscale image to keep only bright pixels, then
# inspect each connected blob's shape.

def detect_stickers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Keep only pixels brighter than the threshold → a binary image.
    _, binary = cv2.threshold(gray, STICKER_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Close to merge the dark gaps inside barcodes/QR codes so a sticker
    # registers as a single blob instead of many stripes.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (STICKER_CLOSE_KERNEL, STICKER_CLOSE_KERNEL)
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # findContours = outline every white blob. We then filter by shape.
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stickers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < STICKER_MIN_AREA or area > STICKER_MAX_AREA:
            continue

        # minAreaRect fits the tightest possibly-rotated rectangle around
        # the blob. A real sticker fills most of that rectangle.
        (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
        if rw == 0 or rh == 0:
            continue
        rect_fit = area / (rw * rh)
        if rect_fit < STICKER_MIN_RECT_FIT:
            continue

        # Reject square-ish blobs — real stickers are noticeably elongated.
        long_side, short_side = max(rw, rh), min(rw, rh)
        if long_side / short_side < STICKER_MIN_ASPECT_RATIO:
            continue

        # Store an axis-aligned bbox; later stages use it for dedupe.
        x, y, w, h = cv2.boundingRect(contour)
        stickers.append((x, y, w, h))
    return stickers


def _point_in_bbox(px, py, bbox):
    x, y, w, h = bbox
    return x <= px <= x + w and y <= py <= y + h


# ===========================================================================
# STAGE 3 — Puck detection
# ===========================================================================
# Pucks are small, bright, round-ish fasteners on the dark panel. We run two
# complementary detectors and merge their results, deduplicating against the
# sticker bboxes from STAGE 2 along the way.

def detect_pucks(image, stickers, panel_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Pass A — Hough Circle Transform.
    # Blur first: HoughCircles is sensitive to noise on the edges it traces.
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

    results = []
    if circles is not None:
        for cx, cy, cr in circles[0]:
            cx, cy, cr = int(cx), int(cy), int(cr)
            if _accept_puck_candidate(gray, cx, cy, cr, stickers, panel_mask):
                results.append((cx, cy, cr))

    # Pass B — ellipse fit on bright contours, for pucks Hough missed.
    results.extend(_detect_pucks_ellipse(gray, results, stickers, panel_mask))
    return results


def _accept_puck_candidate(gray, cx, cy, cr, stickers, panel_mask):
    """Shared sanity checks for both puck-detection passes.

    A puck must be inside the panel mask, outside every sticker bbox, and
    look like a bright disc on a dark background.
    """
    # Filter 1 — must sit on the dark puck panel.
    if not _in_panel(panel_mask, cx, cy):
        return False

    # Filter 2 — STAGE 2 dedupe: drop if the center is inside a sticker.
    if any(_point_in_bbox(cx, cy, s) for s in stickers):
        return False

    # Filter 3 — inner disc must be bright (a real puck is near-white)…
    center_mask = np.zeros_like(gray)
    cv2.circle(center_mask, (cx, cy), max(int(cr * 0.5), 1), 255, -1)
    center_brightness = cv2.mean(gray, mask=center_mask)[0]
    if center_brightness < HOUGH_MIN_CENTER_BRIGHTNESS:
        return False
    # …and contain at least one near-white pixel (dirt never gets that bright).
    _, max_val, _, _ = cv2.minMaxLoc(gray, mask=center_mask)
    if max_val < HOUGH_MIN_CENTER_PEAK:
        return False

    # Filter 4 — surrounding ring must be dark in absolute terms, OR much
    # darker than the center (rescues pucks on over-exposed panels).
    ring_mask = np.zeros_like(gray)
    cv2.circle(ring_mask, (cx, cy), int(cr * 2.0), 255, -1)
    cv2.circle(ring_mask, (cx, cy), int(cr * 1.3), 0, -1)
    if cv2.countNonZero(ring_mask) == 0:
        return False
    surrounding_brightness = cv2.mean(gray, mask=ring_mask)[0]
    if (surrounding_brightness > HOUGH_MAX_SURROUNDING_BRIGHTNESS
            and center_brightness - surrounding_brightness < HOUGH_MIN_CONTRAST):
        return False

    return True


def _detect_pucks_ellipse(gray, existing_pucks, stickers, panel_mask):
    """Fallback puck detector for perspective-distorted (oval) pucks.

    Threshold the image to its bright pixels, fit an ellipse to each blob,
    then keep blobs that look puck-shaped (small, round-ish, well-filled).
    """
    _, binary = cv2.threshold(gray, ELLIPSE_THRESHOLD, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    extras = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < ELLIPSE_MIN_AREA or area > ELLIPSE_MAX_AREA:
            continue
        if len(contour) < 5:  # fitEllipse needs at least 5 boundary points
            continue

        (ex, ey), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        major, minor = max(axis_a, axis_b) / 2, min(axis_a, axis_b) / 2

        # Shape filters: reject blobs that aren't puck-sized, are too oval,
        # or don't fill their fitted ellipse well (irregular shapes).
        if minor < ELLIPSE_MIN_AXIS or major > ELLIPSE_MAX_AXIS:
            continue
        if major / minor > ELLIPSE_MAX_ASPECT:
            continue
        if area / (np.pi * major * minor) < ELLIPSE_MIN_FILL:
            continue

        cx, cy, cr = int(ex), int(ey), int(major)

        # Drop duplicates already found by Hough.
        if any((cx - hx) ** 2 + (cy - hy) ** 2 < HOUGH_MIN_DIST ** 2
               for hx, hy, _ in existing_pucks):
            continue

        # Reuse the shared filters (panel, sticker dedupe, ring darkness).
        if not _accept_puck_candidate(gray, cx, cy, cr, stickers, panel_mask):
            continue

        extras.append((cx, cy, cr))
    return extras


# ===========================================================================
# STAGE 4 — Visualization
# ===========================================================================

def draw_detections(image, pucks, stickers, panel_mask):
    output = image.copy()

    # Tint the panel area green so the mask is visible during tuning.
    if PANEL_DEBUG_OVERLAY and panel_mask is not None:
        tint = np.zeros_like(output)
        tint[:] = (0, 255, 0)
        masked_tint = cv2.bitwise_and(tint, tint, mask=panel_mask)
        output = cv2.addWeighted(output, 1.0, masked_tint, 0.25, 0)

    # Yellow boxes for stickers.
    for x, y, w, h in stickers:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(output, "sticker", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Red circles for pucks, numbered.
    for i, (x, y, r) in enumerate(pucks, start=1):
        cv2.circle(output, (x, y), r, (0, 0, 255), 2)
        cv2.circle(output, (x, y), 1, (0, 0, 255), 2)
        cv2.putText(output, f"#{i}", (x + r + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return output


# ===========================================================================
# Driver — runs all four stages per image, per directory
# ===========================================================================

def process_image(image_path, output_dir):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  skipped (could not read)")
        return

    # The pipeline, top to bottom:
    panel_mask = build_panel_mask(image)               # STAGE 1
    stickers   = detect_stickers(image)                # STAGE 2
    pucks      = detect_pucks(image, stickers, panel_mask)  # STAGE 3
    output     = draw_detections(image, pucks, stickers, panel_mask)  # STAGE 4

    output_path = output_dir / f"{image_path.stem}.jpg"
    cv2.imwrite(str(output_path), output)
    cv2.imwrite(str(output_dir / f"{image_path.stem}_mask.png"), panel_mask)
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
