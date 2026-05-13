# DAF Vision - Puck Detector API

Local HTTP service that takes a single image and returns the pucks detected on it. No authentication, no networking concerns - it binds to `127.0.0.1` only and is intended to be called from another process on the same machine.

The detection itself is classical OpenCV (no machine learning); see `main.py` for the pipeline.

---

## First-time setup

Python 3.12 is required. Run these commands once in the project root.

**Windows (PowerShell or CMD)**

```bat
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**Linux / macOS**

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

---

## Starting the service

Run from the project root. Leave the terminal open; the server runs in the foreground.

**Windows (PowerShell or CMD)**

```bat
.venv\Scripts\uvicorn.exe api:app --host 127.0.0.1 --port 8000
```

**Linux / macOS**

```bash
.venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000
```

The service is ready when the log shows:

```
Uvicorn running on http://127.0.0.1:8000
```

Stop it with `Ctrl+C`.

Interactive API explorer (auto-generated): `http://127.0.0.1:8000/docs`

---

## Running it on boot (production)

`uvicorn` does not daemonize itself. Wrap it with whatever process supervisor the host machine uses so it starts on boot and restarts on crash.

**Windows - using NSSM (Non-Sucking Service Manager)**

```bat
nssm install DafVisionApi "C:\path\to\DAF-Vision\.venv\Scripts\uvicorn.exe" "api:app --host 127.0.0.1 --port 8000"
nssm set DafVisionApi AppDirectory "C:\path\to\DAF-Vision"
nssm start DafVisionApi
```

Alternatives on Windows: Task Scheduler ("At system startup") or a Windows Service via `pywin32`.

**Linux - using systemd**

Create `/etc/systemd/system/daf-vision.service`:

```ini
[Unit]
Description=DAF Vision Puck Detector API
After=network.target

[Service]
WorkingDirectory=/path/to/DAF-Vision
ExecStart=/path/to/DAF-Vision/.venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now daf-vision
```

---

## Endpoints

### `GET /health`

Liveness probe. Returns 200 with model info when the service is up.

```bash
curl http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "ok",
  "model": {
    "type": "opencv-classical",
    "version": "1.0.0",
    "pipeline": ["panel_mask", "sticker_detection", "puck_detection"]
  }
}
```

### `POST /detect`

Accepts one image, returns the pucks and stickers detected in it.

**Request**

| | |
|---|---|
| Method | `POST` |
| URL | `http://127.0.0.1:8000/detect` |
| Content-Type | `multipart/form-data` |
| Form field name | `image` (required) |

One image per request. The service does not accept multiple files in one call.

The service assumes the calling app sends a valid, decodable image (JPEG or PNG). No format, size, or content checks are performed.

Example with `curl`:

```bash
curl -X POST -F "image=@/path/to/panel.jpg" http://127.0.0.1:8000/detect
```

**Response - success (200)**

```json
{
  "puck_count": 14,
  "pucks": [
    { "id": 1, "x": 840, "y": 643, "radius": 9 },
    { "id": 2, "x": 475, "y": 654, "radius": 9 }
  ],
  "stickers": [
    { "x": 220, "y": 180, "width": 60, "height": 25 }
  ],
  "image": { "width": 1920, "height": 1080 },
  "model": {
    "type": "opencv-classical",
    "version": "1.0.0",
    "pipeline": ["panel_mask", "sticker_detection", "puck_detection"]
  },
  "processing_time_ms": 142
}
```

---

## Response fields explained

| Field | Type | Meaning |
|---|---|---|
| `puck_count` | int | Number of pucks detected. Same as `pucks.length`. The headline number the consumer most often needs. |
| `pucks` | array | One entry per detected puck. |
| `pucks[].id` | int | 1-based index in detection order. Matches the `#N` label that the offline pipeline draws on annotated debug images. Stable within a single response only - do not treat as a persistent identifier across calls. |
| `pucks[].x` | int | Center X coordinate in image pixels. Origin is the top-left corner. |
| `pucks[].y` | int | Center Y coordinate in image pixels. Origin is the top-left corner. |
| `pucks[].radius` | int | Approximate puck radius in pixels. Useful for drawing overlays, less so for measurements. |
| `stickers` | array | Bright rectangular labels (barcodes, QR stickers) detected on the panel. Returned for diagnostic purposes - the detector uses them internally to avoid counting them as pucks. Most consumers can ignore this field. |
| `stickers[].x` | int | Top-left X of the sticker bounding box. |
| `stickers[].y` | int | Top-left Y of the sticker bounding box. |
| `stickers[].width` | int | Width of the bounding box in pixels. |
| `stickers[].height` | int | Height of the bounding box in pixels. |
| `image.width` | int | Width of the input image as decoded. |
| `image.height` | int | Height of the input image as decoded. |
| `model.type` | string | Always `"opencv-classical"`. Indicates this is a deterministic CV pipeline, not a machine-learning model. |
| `model.version` | string | Service version. Bump this when detection behavior changes meaningfully. |
| `model.pipeline` | string[] | Names of the stages run, in order. Informational. |
| `processing_time_ms` | int | Wall-clock time spent in the CV pipeline only - does not include network or upload time. |

Coordinate system reminder: `(0, 0)` is the **top-left** of the image, X increases to the right, Y increases downward. This matches OpenCV and most image libraries (including `System.Drawing` in .NET).

---

## Error responses

The service performs no validation on the uploaded image. Any failure (missing form field, unreadable bytes, internal error) results in a `422` or `500` response with FastAPI's default error body. Consumers should treat any non-200 response as a failed detection.

---

## Behavior notes for the calling app

- **One image per request, one request at a time.** The service is sized for sequential calls. Concurrent requests will queue and run serially.
- **No persistence.** The image is read into memory, processed, and discarded. The service writes no files.
- **No state between calls.** Each request is independent.
- **`puck_count == 0` is a valid success response.** A clean panel with no pucks returns 200 with an empty `pucks` array - not an error.
- **The `id` field is per-response only.** If the same panel is photographed twice and sent in two separate requests, a given puck may get a different `id`. Use `(x, y)` if you need to correlate detections to physical positions.
- **`processing_time_ms` excludes upload time.** If the calling app needs end-to-end latency, measure it on the client side.
- **`stickers` is informational.** Whether to display or ignore it is the consumer's choice; the puck detector has already accounted for them.
