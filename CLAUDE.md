# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the FastAPI server (development)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Docker (production)
docker compose up -d
docker compose logs -f
docker compose down

# Syntax-check tools without running them
python -m py_compile tools/prepare_yolo_dataset.py tools/cosmos_predict25_batch.py
docker compose config

# Generate a YOLO dataset from Label Studio
python tools/prepare_yolo_dataset.py -p PROJECT_ID -o ./datasets/my_project --auto-label --force

# Full balance loop (Label Studio + Cosmos + SAM3)
python tools/prepare_yolo_dataset.py -p 1 --use-existing --auto-label \
  -o ./datasets/my_project --save-preview --balance-classes --cosmos-augment \
  --max-workers --sam3-batch-size auto --force
```

## Architecture

### FastAPI server (`app/`)

`app/main.py` is the single-file FastAPI application. All routes live here. The SAM3 detector is loaded once at startup via `@lru_cache()` on `_get_detector()` and injected as a FastAPI dependency.

Key routes:
- `POST /detect` — RTSP stream detection
- `GET /live-stream` — MJPEG stream with live annotations
- `POST /predict` and `POST /label-studio/predict` — Label Studio ML backend
- `POST /setup` and `POST /refresh-labels` — clear the Label Studio label cache
- `GET /labels` — inspect cached labels

**Configuration** (`app/core/config.py`): Pydantic `BaseSettings` with `SAM3_` env prefix. All settings load from `.env`. `get_settings()` is also `@lru_cache()`-ed — call `get_settings.cache_clear()` if you need to reload settings in tests.

**Label Studio integration** (`app/services/labelstudio.py`): Labels are fetched from the LS API and cached in-process for 1 hour. The cache is cleared on every `/setup` call (which LS sends when the ML backend is registered). The `/predict` handler uses these cached labels to filter SAM3 output — only labels that exist in the LS project config are returned.

### SAM3 detector (`app/models/sam3_detector.py`)

`Sam3Detector` wraps the SAM3 model. Detection flow per frame:
1. `set_image()` — expensive ViT encoding, done once per frame
2. Per-concept `set_text_prompt()` loop — cheap text queries
3. Per-class NMS → cross-class NMS → `max_detections` cap

The `sam3.sam` submodule is absent from the published PyPI wheel. `_extend_sam3_namespace()` patches it in at import time from `third_party/sam3_patch/`. The BPE vocabulary file is located by searching `sam3`, `clip`, and `open_clip` packages in that order.

The label mapping dict in `_format_label_studio_results()` (in `app/main.py`) normalizes SAM3 output labels (e.g. `"lorry"` → `"truck"`) to match LS project labels before returning predictions.

### Tools

**`tools/prepare_yolo_dataset.py`** — standalone CLI that talks to Label Studio and the SAM3 `/predict` endpoint. Streams tasks in batches, downloads images in parallel, converts `RectangleLabels` annotations to YOLO format, splits into train/val/test, and writes `dataset.yaml`. With `--balance-classes --cosmos-augment` it also invokes LM Studio for prompt generation and triggers Cosmos jobs.

**`tools/cosmos_predict25_batch.py`** — creates and optionally runs NVIDIA Cosmos Predict2.5 generation jobs. Cosmos runs as a separate Docker service (`cosmos` in `docker-compose.yml`) built from a sibling repo checkout at `../cosmos-predict2.5`. Synthetic images are always placed in `train/` only, never `val/` or `test/`.

### Docker services

`docker-compose.yml` defines two services:
- `sam3-auto-labeler` — the FastAPI server, mounts `./weights` and `./datasets`
- `cosmos` (optional) — Cosmos Predict2.5, expects `../cosmos-predict2.5` checked out as a sibling directory; communicates with the labeler via the `/auto-labeler` volume mount

## Environment

Copy `.env.example` to `.env`. Minimum required:

```env
SAM3_HF_TOKEN=your_huggingface_token   # needed to download the ~2GB SAM3 checkpoint
SAM3_DEVICE=cuda                        # or cpu
```

Label Studio integration requires `SAM3_LABELSTUDIO_API_BASE` and `SAM3_LABELSTUDIO_API_TOKEN`. The model checkpoint is downloaded automatically from HuggingFace on first run if `SAM3_CHECKPOINT_PATH` is not set or the file doesn't exist.

## Key constraints

- `sam3` must be installed from source: `sam3 @ git+https://github.com/facebookresearch/sam3.git` (see `requirements.txt`)
- PyTorch must be installed separately before `pip install -r requirements.txt` because it requires a CUDA-specific index URL
- The `third_party/sam3_patch/` directory vendors the missing `sam3.sam` subpackage — do not remove it
- Cosmos Predict2.5 requires a second GPU (configured via `COSMOS_NVIDIA_VISIBLE_DEVICES`); the SAM3 service defaults to GPU 0
