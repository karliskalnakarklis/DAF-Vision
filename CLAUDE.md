# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

```bash
.venv/bin/python main.py
```

This processes every image in each subdirectory of `images/` (e.g. `images/clean/`, `images/defects/`) and writes annotated outputs plus a `<name>_mask.png` panel mask into the matching subdirectory of `output/`. There is no separate build, test, or lint step — the project is a single script.

Dependencies (OpenCV + NumPy) live in `.venv/` (Python 3.12). `images/` and `output/` are gitignored, so the working dataset is local-only.

## Architecture

`main.py` is a four-stage classical-CV pipeline (no ML). Each image flows through the stages in order, and each later stage trusts the earlier stages' outputs:

1. **Panel mask** (`build_panel_mask` + `_build_frame_mask`) — produces a binary mask of the dark underbody panel. Built from grayscale + HSV-saturation thresholding, then morphological close/open, largest-component selection, contour fill, and subtraction of a separately-built "side frame" mask. A `_gray_world_balance` pre-step neutralizes color casts so the saturation thresholds work across lighting conditions. Output: a uint8 mask where only legitimate puck territory is white.

2. **Sticker detection** (`detect_stickers`) — binary-thresholds the grayscale image to find bright blobs, then filters by `minAreaRect` fit and aspect ratio to keep only elongated, rectangular labels. Output: a list of `(x, y, w, h)` bboxes that later stages exclude.

3. **Puck detection** (`detect_pucks`) — two complementary passes whose results are merged:
   - **Pass A**: `cv2.HoughCircles` on a Gaussian-blurred grayscale (fast, finds well-formed circular pucks).
   - **Pass B**: `_detect_pucks_ellipse` fits ellipses to bright contours (catches perspective-distorted pucks Hough misses), deduped against Pass A by squared-distance vs. `HOUGH_MIN_DIST`.
   
   Both passes funnel candidates through `_accept_puck_candidate`, which enforces: inside panel mask, outside every sticker bbox, bright disc center (mean + peak), and a dark surrounding ring (absolute or relative to center for over-exposed panels).

4. **Visualization** (`draw_detections`) — draws yellow sticker boxes and numbered red puck circles.

The driver (`main` → `process_directory` → `process_image`) just walks the directory tree and applies the four stages.

## Tuning

All thresholds live as module-level constants at the top of `main.py`, grouped by stage (`PANEL_*`, `FRAME_*`, `STICKER_*`, `HOUGH_*`, `ELLIPSE_*`). When detection misbehaves, the fix is almost always tweaking these — not restructuring code. The per-image `<name>_mask.png` file is the primary visual debugging tool. Each constant has an inline comment explaining what it controls and which direction loosens vs. tightens the filter.
