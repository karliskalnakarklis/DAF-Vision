"""Probe why the bottom-right puck is missing in F8090_image_1_20260511_212919.

Strategy: read the input image, run the panel mask + sticker stages exactly
like main.py, then look at the bottom-right area for any bright blob and
walk it through _accept_puck_candidate one filter at a time, printing which
filter (if any) rejects it. Also print what HoughCircles + the ellipse fit
actually return near that area, so we can see whether the candidate is even
being proposed by either pass.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import main as m

IMG = Path(__file__).parent / "images/clean/F8071_image_1_20260511_193840.jpg"

image = cv2.imread(str(IMG))
H, W = image.shape[:2]
print(f"image {W}x{H}")

panel_mask = m.build_panel_mask(image)
stickers = m.detect_stickers(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Search the top-right quadrant for the brightest blob (the missing puck).
roi_x0, roi_y0 = int(W * 0.85), 0
roi_x1, roi_y1 = W, int(H * 0.25)
roi = gray[roi_y0:roi_y1, roi_x0:roi_x1]
_, binary = cv2.threshold(roi, 130, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"bright blobs in top-right ROI ({roi_x0},{roi_y0})-({roi_x1},{roi_y1}): {len(contours)}")

candidates = []
for c in contours:
    area = cv2.contourArea(c)
    if area < 5:
        continue
    M = cv2.moments(c)
    if M["m00"] == 0:
        continue
    cx = int(M["m10"] / M["m00"]) + roi_x0
    cy = int(M["m01"] / M["m00"]) + roi_y0
    candidates.append((cx, cy, area, c))

# Pick the candidate inside the panel mask (the actual puck), not the
# largest blob (which is the bright chassis bit in the very corner).
candidates = [t for t in candidates if panel_mask[t[1], t[0]] != 0] or candidates
candidates.sort(key=lambda t: -t[2])
print(f"\ntop bright blobs in bottom-right (by area):")
for cx, cy, area, _ in candidates[:5]:
    in_panel = panel_mask[cy, cx] != 0 if (0 <= cy < H and 0 <= cx < W) else False
    print(f"  ({cx:4d},{cy:4d})  area={area:6.0f}  panel_mask={'WHITE' if in_panel else 'BLACK'}")

# Pick the largest as the candidate puck and walk it through every filter.
if not candidates:
    print("no bright blobs found in bottom-right ROI")
    raise SystemExit
cx, cy, area, contour = candidates[0]
print(f"\n=== probing brightest blob at ({cx},{cy}), area={area} ===")

# Shift the contour from ROI-local back to full-image coordinates before
# fitting, so the printed ellipse center makes sense.
contour_full = contour + np.array([[roi_x0, roi_y0]])

if len(contour_full) >= 5:
    (ex, ey), (a, b), _ = cv2.fitEllipse(contour_full)
    major = max(a, b) / 2
    minor = min(a, b) / 2
    cr = int(major)
    print(f"ellipse fit: center=({ex:.1f},{ey:.1f}) major={major:.1f} minor={minor:.1f} aspect={major/minor:.2f}")
else:
    cr = 10
    print(f"contour too small to fitEllipse, using cr={cr}")

print(f"\n--- _accept_puck_candidate filter-by-filter ---")
# Filter 1
in_panel = m._in_panel(panel_mask, cx, cy)
print(f"  F1 in_panel({cx},{cy})           : {in_panel}")

# Filter 2
in_sticker = any(m._point_in_bbox(cx, cy, s) for s in stickers)
print(f"  F2 inside a sticker bbox          : {in_sticker}")

# Filter 3a/b
center_mask = np.zeros_like(gray)
cv2.circle(center_mask, (cx, cy), max(int(cr * 0.5), 1), 255, -1)
center_brightness = cv2.mean(gray, mask=center_mask)[0]
_, max_val, _, _ = cv2.minMaxLoc(gray, mask=center_mask)
print(f"  F3 center mean brightness         : {center_brightness:.1f}  (need >= {m.HOUGH_MIN_CENTER_BRIGHTNESS})")
print(f"  F3 center peak brightness         : {max_val:.1f}  (need >= {m.HOUGH_MIN_CENTER_PEAK})")

# Filter 4
ring_mask = np.zeros_like(gray)
cv2.circle(ring_mask, (cx, cy), int(cr * 2.0), 255, -1)
cv2.circle(ring_mask, (cx, cy), int(cr * 1.3), 0, -1)
ring_brightness = cv2.mean(gray, mask=ring_mask)[0] if cv2.countNonZero(ring_mask) else None
contrast = center_brightness - (ring_brightness or 0)
print(f"  F4 ring  mean brightness         : {ring_brightness:.1f}  (need <= {m.HOUGH_MAX_SURROUNDING_BRIGHTNESS}, OR contrast >= {m.HOUGH_MIN_CONTRAST})")
print(f"  F4 contrast (center - ring)      : {contrast:.1f}")

result = m._accept_puck_candidate(gray, cx, cy, cr, stickers, panel_mask)
print(f"\nFINAL _accept_puck_candidate -> {result}")

# Also: does the ellipse pass even propose this candidate?
print(f"\n--- ellipse-pass shape filters on this contour ---")
if len(contour) >= 5:
    print(f"  area                       : {area:.0f}  (need in [{m.ELLIPSE_MIN_AREA}, {m.ELLIPSE_MAX_AREA}])")
    print(f"  minor radius               : {minor:.1f} (need >= {m.ELLIPSE_MIN_AXIS})")
    print(f"  major radius               : {major:.1f} (need <= {m.ELLIPSE_MAX_AXIS})")
    print(f"  major/minor                : {major/minor:.2f} (need <= {m.ELLIPSE_MAX_ASPECT})")
    fill = area / (np.pi * major * minor)
    print(f"  fill                       : {fill:.2f} (need >= {m.ELLIPSE_MIN_FILL})")
