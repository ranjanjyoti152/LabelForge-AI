# Cosmos Predict2.5 Dataset Balancing Plan

This project keeps SAM3 as the only labeling engine. NVIDIA Cosmos Predict2.5 is used only to generate extra visual samples when Label Studio does not contain enough examples for one or more classes.

## Target Flow

```text
Label Studio project
  -> fetch real images and existing annotations
  -> SAM3 labels missing real annotations
  -> count samples per class
  -> identify weak classes below DATASET_MIN_SAMPLES_PER_CLASS
  -> LM Studio creates project-specific Cosmos prompts for weak classes
  -> Cosmos Predict2.5 generates class-targeted videos
  -> extract frames from generated videos
  -> SAM3 labels generated frames
  -> filter low-quality synthetic samples
  -> merge accepted synthetic samples into train/ only
  -> write dataset.yaml and quality_report.json
```

## Docker Layout

`docker-compose.yml` now has two roles:

- `sam3-auto-labeler`: long-running FastAPI service on port `8000`.
- `cosmos`: batch generation service from the external `nvidia-cosmos/cosmos-predict2.5` checkout.

The `cosmos` service depends on `sam3-auto-labeler`, so `docker compose up --build` starts both containers. Cosmos is not included in the SAM3 image because it is large, has different CUDA/runtime requirements, and is best used as a batch generator.

## Setup

Clone Cosmos next to this repository:

```bash
cd /home/proxpc
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git
cd cosmos-predict2.5
git lfs pull
```

For the current `cu128` Docker build, keep the Cosmos checkout on Python 3.10:

```bash
printf "3.10\n" > .python-version
```

The `flash-attn` CUDA wheel resolved by the Cosmos lockfile is published for the `cp310` ABI in this configuration.

Set Hugging Face access:

```bash
export HF_TOKEN=your_huggingface_read_token
```

Build and start SAM3 plus Cosmos:

```bash
cd /home/proxpc/LabelForge-AI
docker compose up -d --build
```

Check that the container starts:

```bash
docker compose run --rm cosmos python examples/inference.py --help
```

## Batch Generation

The fully automatic path is now part of `tools/prepare_yolo_dataset.py`. LM Studio is optional but recommended because every Label Studio project has different class names and visual context.

Start the LM Studio local server, load a prompt-capable model, then set:

```bash
LM_STUDIO_API_BASE=http://localhost:1234/v1
LM_STUDIO_MODEL=your-loaded-model-id
LM_STUDIO_API_KEY=lm-studio
```

Run the complete loop:

```bash
python tools/prepare_yolo_dataset.py \
  -p 64 \
  --use-existing \
  --auto-label \
  -o ./datasets/my_project \
  --balance-classes \
  --cosmos-augment \
  --min-samples-per-class 300 \
  --force
```

The script will:

- count real boxes per class,
- ask LM Studio for one Cosmos prompt per weak class,
- run Cosmos through Docker,
- extract frames,
- label generated frames with SAM3,
- copy accepted samples into `train/` only,
- write `quality_report.json`.

You can also run only the standalone Cosmos wrapper if you already have a prompt JSON file.

Create generation jobs from a JSON prompt file:

```bash
python tools/cosmos_predict25_batch.py \
  --prompts prompts/cosmos_class_prompts.example.json
```

Build, run, and extract frames:

```bash
python tools/cosmos_predict25_batch.py \
  --prompts prompts/cosmos_class_prompts.example.json \
  --build \
  --run \
  --extract-frames
```

The wrapper writes:

```text
cosmos/jobs/       # generated Cosmos input JSON files
cosmos/outputs/    # generated videos
cosmos/frames/     # sampled frames for SAM3 labeling
```

These generated directories are ignored by git.

## Quality Policy

Synthetic data must be conservative:

- Use Cosmos only for classes below `DATASET_MIN_SAMPLES_PER_CLASS`.
- Keep generated samples in `train/` only.
- Never place synthetic images in `val/` or `test/`.
- Accept a synthetic frame only if SAM3 detects the weak class.
- Reject frames with no target class, tiny boxes, excessive duplicate boxes, or low-confidence detections.
- Enforce `COSMOS_SYNTHETIC_MAX_RATIO` so synthetic samples never dominate real samples.
