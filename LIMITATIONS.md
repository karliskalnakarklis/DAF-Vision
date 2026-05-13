# Limitations of the Classical-CV Approach

## What this pipeline is

`main.py` is a four-stage classical-CV pipeline (panel mask → sticker detection → puck detection → visualization) built on hand-tuned thresholds. There is no learned model — every decision is a fixed numeric rule:

- **Panel mask**: a pixel is "panel" if grayscale `< 75` *and* HSV saturation `< 110`.
- **Frame exclusion**: a pixel is "frame" if grayscale `> 110` *or* saturation `> 140`.
- **Stickers**: bright blobs (`gray > 160`) with rectangle-fit `≥ 0.80` and aspect ratio `≥ 1.3`.
- **Pucks**: Hough circles + ellipse fits, gated by center brightness `≥ 110`, peak `≥ 160`, ring brightness `≤ 70` (or contrast `≥ 60`).

These constants are not learned from data — they were chosen by looking at the current sample images. The `_gray_world_balance` step does soften per-image color casts, but it cannot rescue a fundamentally different exposure or viewpoint.

## Why it's brittle

Every threshold above encodes an assumption about the scene:

| Assumption | What breaks it |
|---|---|
| Panel is darker than 75/255 grayscale | Brighter ambient light, sunlight through a door, a new fixture lamp |
| Frame is brighter than 110/255 | Shadow falling on the frame, dirt or grime on side rails |
| Pucks have a peak pixel ≥ 160 | Dim lighting, oblique angle reducing specular highlight |
| Pucks are 6–16 px radius (Hough) | Camera moved closer/farther, different focal length, different resolution |
| Frame touches the left/right image edge | Camera rotated, panel re-cropped, framing shifted |
| Stickers are 120–4000 px² and elongated | Different sticker stock, closer camera making them larger, rotated label |

These are *coupled*: a single change to camera distance shifts puck radius, sticker area, *and* the spatial relationship between panel and frame edges all at once. There is no single knob to "scale for new conditions" — every constant has to be re-tuned together.

## Could the code itself be better?

Marginally, yes — but not in ways that solve the real problem:

- **Adaptive thresholding** (`cv2.adaptiveThreshold`, Otsu) instead of fixed thresholds would handle global brightness shifts somewhat better. Won't help with viewpoint or partial shadows.
- **Normalize by panel statistics**: after building the panel mask, recompute puck-detection thresholds as offsets from the panel's own mean/std brightness. More robust to overall exposure, still fragile to local glare.
- **Auto-tune constants**: a small script that grid-searches the constants against a labeled set would replace manual tuning. Useful, but only as good as the labels and only valid for the conditions in that label set.

These would reduce sensitivity, not remove it. The pipeline is structurally tied to its assumptions about the scene.

## The honest assessment

**You are right.** With a classical-CV pipeline of this shape, the dominant cost of false positives/negatives is *not* the code — it is variance in the input. The biggest single improvement available to you is removing that variance at the source. In rough order of leverage:

1. **Fixed camera mount and fixed lighting.** Same position, same focal length, same exposure, same lamps every time. This is what industrial vision lines do, and the reason they get away with classical CV at all. Diffuse lighting (a softbox or LED panel rather than a bare bulb) also kills the specular hotspots that confuse the bright/dark logic.
2. **Controlled background / fixture.** If the truck always sits in the same jig, the panel is in the same image coordinates every time — you can hard-code an ROI and skip the entire panel-mask stage, which is the most failure-prone part of the pipeline.
3. **Calibration target in frame.** A small known patch (a gray card or ArUco marker) in every shot lets the pipeline auto-correct exposure and detect framing drift instead of silently misclassifying.

If 1–3 are achievable, this codebase will likely meet your needs after a single re-tune. If they are not — if the camera will be handheld, or lighting will vary, or the truck position will shift — then **the next step is not better classical CV, it is a learned detector** (e.g. a small YOLO or a segmentation model trained on a few hundred labeled images). A learned model handles the "everything changes a little at once" case that thresholds cannot.

## Recommendation

1. **First, fix the capture environment.** Mount the camera, fix the lighting, and add a gray card or ArUco marker for calibration. Re-evaluate accuracy *after* doing this — you may discover the current pipeline is now sufficient.
2. **If the capture environment cannot be fixed**, accept that classical CV is the wrong tool and move to a learned detector. A few hundred labeled images and a fine-tuned small detector will outperform any further threshold tuning, and will degrade gracefully instead of failing sharply.
3. **Either way, build a labeled evaluation set** (images + ground-truth puck/sticker locations) before changing anything else. Without it, you cannot tell whether a tweak is improving the pipeline or just shifting the failure cases around.

The code is about as good as a hand-tuned classical pipeline of this shape can reasonably be. Further engineering effort spent inside `main.py` will have diminishing returns compared to either of the two paths above.
