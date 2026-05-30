# Contributing to LabelForge AI

Thanks for helping improve LabelForge AI. The project is most valuable when it stays reliable for real dataset work: Label Studio import, SAM3 auto-labeling, YOLO export, and conservative Cosmos augmentation.

## Good Contributions

- Fix reproducible bugs with clear logs or screenshots.
- Improve Docker, CUDA, or dependency setup.
- Add tests or validation scripts for dataset conversion.
- Improve Label Studio, SAM3, LM Studio, or Cosmos integration.
- Add docs for real hardware, GPU, and dataset workflows.
- Improve safety filters for synthetic data quality.

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/LabelForge-AI.git
cd LabelForge-AI
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

For Docker validation:

```bash
docker compose config
docker compose up -d --build
```

## Branches and Commits

Use short, descriptive branches:

```bash
git checkout -b feature/cosmos-quality-filter
git checkout -b fix/labelstudio-url-handling
git checkout -b docs/docker-cuda-setup
```

Keep commits focused and use clear messages:

```bash
git commit -m "Add Cosmos synthetic frame filtering"
```

## Validation Checklist

Before opening a pull request, run the checks that match your change:

```bash
python -m py_compile tools/prepare_yolo_dataset.py tools/cosmos_predict25_batch.py
docker compose config
```

For dataset pipeline changes, include:

- the command you ran,
- Label Studio project size or a small sample description,
- whether `--use-existing`, `--auto-label`, or `--cosmos-augment` was used,
- resulting train/val/test counts,
- any rejected synthetic frame reasons from `quality_report.json`.

## Synthetic Data Rules

Cosmos-generated samples are useful only when they improve training quality. Please keep these rules intact:

- SAM3 remains the labeling engine.
- Cosmos samples are added only for weak classes.
- Synthetic images go into `train/` only.
- Synthetic images must never be copied into `val/` or `test/`.
- Generated frames are accepted only when SAM3 detects the target class.
- `quality_report.json` should explain what was generated and accepted.

## Pull Request Checklist

- The change is focused and avoids unrelated refactors.
- README or docs are updated for user-facing behavior.
- New configuration appears in `.env.example` and `.env.reference`.
- Secrets are not committed.
- Validation commands and results are included in the PR body.

## Security

Do not commit API keys, Hugging Face tokens, Label Studio tokens, model weights, generated datasets, or private images. Use `.env` for secrets; it is intentionally git-ignored.
