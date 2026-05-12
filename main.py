import cv2
import numpy as np
from pathlib import Path


IMAGES_DIR = Path(__file__).parent / "images"
OUTPUT_DIR = Path(__file__).parent / "output"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Panel mask: isolate the dark puck panel so detections elsewhere (side frame,
# chassis cutouts, scaffolding behind the truck) can be rejected.
PANEL_BRIGHTNESS_MAX = 75   # pixel is "panel" if grayscale value < this
PANEL_SATURATION_MAX = 110  # AND HSV saturation < this (panel is desaturated)
PANEL_CLOSE_KERNEL = 71     # large enough to bridge weld seams + fill puck-size bright holes
PANEL_OPEN_KERNEL = 35      # break narrow corridors so dark corner shoulders don't sneak in
PANEL_ERODE = 6             # small final erosion to keep mask off the frame transition
PANEL_MIN_AREA = 50000      # ignore tiny dark blobs; the real panel is huge
PANEL_DEBUG_OVERLAY = True  # draw the mask on output for visual tuning

# Frame exclusion: bright/saturated regions touching the left/right border are the truck
# side frame. Dilate generously and subtract from panel so dark "shoulders" at the
# corners (where the frame curves out of the image) don't sneak through.
FRAME_BRIGHTNESS_MIN = 110  # pixel is "frame-like" if brighter than this
FRAME_SATURATION_MIN = 140  # OR more saturated than this (blue/red/green frame)
FRAME_ERODE = 11            # strip out isolated bright spots (pucks) before merging frame
FRAME_CLOSE_KERNEL = 41     # merge surviving frame fragments into one component
FRAME_DILATE = 25           # how far the final frame exclusion reaches into the corners

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


def build_panel_mask(image):
    """Binary mask of the dark puck panel, with bright puck holes filled in.

    Anything outside this mask (side frame, scaffolding, chassis cutouts) is
    not a valid place for a puck.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]

    panel = ((gray < PANEL_BRIGHTNESS_MAX) & (saturation < PANEL_SATURATION_MAX))
    panel = panel.astype(np.uint8) * 255

    # Close to fill in pucks/stickers/seams that sit inside the panel.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (PANEL_CLOSE_KERNEL, PANEL_CLOSE_KERNEL)
    )
    panel = cv2.morphologyEx(panel, cv2.MORPH_CLOSE, kernel)

    # Open to disconnect narrow corridors (e.g. corner shoulders connected via a
    # thin path around the side frame). The shoulders become separate small
    # components and get dropped by the next step.
    if PANEL_OPEN_KERNEL > 0:
        open_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (PANEL_OPEN_KERNEL, PANEL_OPEN_KERNEL)
        )
        panel = cv2.morphologyEx(panel, cv2.MORPH_OPEN, open_k)

    # Keep only the largest connected component (the panel itself).
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(panel, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(panel)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < PANEL_MIN_AREA:
        return np.zeros_like(panel)
    mask = (labels == largest).astype(np.uint8) * 255

    # Fill internal holes (bright weld seams, sealant blobs the close-kernel can't span).
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    # Subtract a dilated frame mask so dark "shoulders" at the corners (where the
    # bright frame curves out of view) don't end up inside the panel.
    frame = _build_frame_mask(gray, saturation)
    filled = cv2.bitwise_and(filled, cv2.bitwise_not(frame))

    # Light erosion to clear the frame transition zone.
    if PANEL_ERODE > 0:
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (PANEL_ERODE * 2 + 1, PANEL_ERODE * 2 + 1)
        )
        filled = cv2.erode(filled, erode_kernel)
    return filled


def _build_frame_mask(gray, saturation):
    """Bright/saturated regions touching the left or right image border, dilated.

    These are the truck side frame and any scaffolding visible through chassis
    cutouts. Dilation extends the exclusion into the corner shoulders.
    """
    frame = ((gray > FRAME_BRIGHTNESS_MIN) | (saturation > FRAME_SATURATION_MIN))
    frame = frame.astype(np.uint8) * 255

    # Erode to wipe out isolated bright spots (pucks). The real frame is thicker
    # than a puck, so it survives this step while pucks disappear entirely.
    if FRAME_ERODE > 0:
        erode_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (FRAME_ERODE * 2 + 1, FRAME_ERODE * 2 + 1)
        )
        frame = cv2.erode(frame, erode_k)

    # Merge frame fragments so the whole side rail is one component.
    close_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (FRAME_CLOSE_KERNEL, FRAME_CLOSE_KERNEL)
    )
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, close_k)

    # Keep only components that touch the left or right border. The central weld
    # seam touches top/bottom but never the sides, so this excludes it.
    num_labels, labels = cv2.connectedComponents(frame)
    edge_labels = set(np.unique(labels[:, 0]).tolist())
    edge_labels.update(np.unique(labels[:, -1]).tolist())
    edge_labels.discard(0)
    if not edge_labels:
        return np.zeros_like(frame)
    side_frame = np.isin(labels, list(edge_labels)).astype(np.uint8) * 255

    # Dilate to reach into the corners and cover the transition zone.
    dilate_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (FRAME_DILATE * 2 + 1, FRAME_DILATE * 2 + 1)
    )
    return cv2.dilate(side_frame, dilate_k)


def _in_panel(panel_mask, cx, cy):
    h, w = panel_mask.shape
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return False
    return panel_mask[cy, cx] != 0


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


def detect_pucks(image, stickers, panel_mask):
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

        # Must sit on the dark puck panel — rejects detections on the side frame.
        if not _in_panel(panel_mask, cx, cy):
            continue

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
    results.extend(_detect_pucks_ellipse(gray, results, stickers, panel_mask))
    return results


def _detect_pucks_ellipse(gray, existing_pucks, stickers, panel_mask):
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

        # Must sit on the dark puck panel.
        if not _in_panel(panel_mask, cx, cy):
            continue

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


def draw_detections(image, pucks, stickers, panel_mask):
    output = image.copy()

    if PANEL_DEBUG_OVERLAY and panel_mask is not None:
        # Tint the panel area green so the mask is visible during tuning.
        tint = np.zeros_like(output)
        tint[:] = (0, 255, 0)
        masked_tint = cv2.bitwise_and(tint, tint, mask=panel_mask)
        output = cv2.addWeighted(output, 1.0, masked_tint, 0.25, 0)

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

    panel_mask = build_panel_mask(image)
    stickers = detect_stickers(image)
    pucks = detect_pucks(image, stickers, panel_mask)
    output = draw_detections(image, pucks, stickers, panel_mask)

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
