# Telegram Tilesheet Bot

This project provides a Telegram bot that builds a tilesheet from user‑uploaded images.
It supports background removal, basic frame normalization (scale by alpha height + baseline
alignment), and optional debug dumps for postprocessing.

## What it does

- `/tilesheet` starts a session with tile size and grid size.
- User uploads images (photo or document). The bot can also split a strip or grid
  when image width/height is a multiple of `tile_width`/`tile_height`.
- Each frame:
  - is converted to RGBA,
  - background is removed with `rembg` unless transparency already exists,
  - normalized by alpha bbox height + baseline (feet) into a fixed tile.
- Frames are assembled left‑to‑right, top‑to‑bottom into a single PNG tilesheet.

## Commands

- `/start` - short help
- `/tilesheet 256x512 5x5` - start a session (tile size + grid)
- `/remove_bg [512x512]` - remove background from a single image (optional resize)
- `/resize 512x512` - resize a single image without background removal
- `/done` - finish early (if fewer images than grid)
- `/cancel` - reset session

## /tilesheet options

Options can be passed as `key=value` or as standalone tokens:

- `padding=0` (default 0)
- `order=time|name` (default time)
- `bg=remove|keep` (default remove)
- `alpha=16` or `alpha_threshold=16` (default 16)
- `margin=8` or `margin_bottom=8` (default 8)
- `clamp=0.85..1.15` (default 0.85..1.15)
- `resample=nearest|bicubic` (default bicubic)

Example:
```
/tilesheet 256x512 5x5 padding=4 order=name bg=remove alpha=16 margin=8 clamp=0.9..1.1 resample=bicubic
```

## Postprocessing logic (2‑pass)

**Pass A (analysis)**
- Convert to RGBA, build alpha mask (alpha > threshold).
- Find bbox (min rect covering all alpha pixels).
- Find baseline as the lowest y with any alpha.
- Compute `content_h = bbox.height` and `baseline_offset = baseline_y - bbox_top`.
- Collect content heights; median is used as `target_content_h` (with outlier filtering).

**Pass B (normalize)**
- Crop to bbox, resize by `target_content_h / content_h` (clamped).
- Compute baseline in resized frame.
- Paste onto a transparent tile:
  - X is centered
  - Y aligns baseline to `tile_h - margin_bottom`
- If paste falls outside tile bounds, it is clipped safely.

## .env configuration

Required:
```
BOT_TOKEN=123:abc
```

Optional:
```
TEMP_DIR=temp
MAX_IMAGES_PER_SESSION=0

READ_TIMEOUT=120
WRITE_TIMEOUT=120
CONNECT_TIMEOUT=20
POOL_TIMEOUT=10

SEND_READ_TIMEOUT=300
SEND_WRITE_TIMEOUT=300
SEND_CONNECT_TIMEOUT=20
SEND_POOL_TIMEOUT=10

DEBUG_POSTPROCESS=0
DEBUG_POSTPROCESS_DIR=temp/debug_postprocess
```

Notes:
- `SEND_*` timeouts are used for `reply_document` upload (large PNGs).
- Debug mode saves per‑frame artifacts: input, cropped, resized, tile, and a baseline line.

## Files and structure

```
bot.py                         Entry point, sets up Telegram app
remove_bg.py                   CLI background remover
tilesheet_bot/
  background_remover.py        rembg wrapper with transparency check
  bot_controller.py            Telegram handlers + session flow
  config.py                    .env reader
  models.py                    TilesheetParams, StoredImage
  postprocess.py               2‑pass normalization module
  session_manager.py           Per‑chat sessions
  storage.py                   Temp file storage
  tilesheet_builder.py         Assembles tilesheet grid
```

## Run

```
python bot.py
```

Make sure `.env` contains `BOT_TOKEN`.

## Docker run

1. Create `.env` (or copy from `.env.example`) and set `BOT_TOKEN`.
2. Build and start:
   ```
   docker compose up --build -d
   ```
3. View logs:
   ```
   docker compose logs -f bot
   ```
4. Stop:
   ```
   docker compose down
   ```

Notes:
- `./temp` is mounted to `/app/temp` for temporary files and debug outputs.
- `rembg-models` volume stores model cache in `/models` (faster restarts).

## CLI background remover

```
python remove_bg.py input.png output.png --width 512 --height 512
```

If the image already has transparency, `rembg` is skipped.
