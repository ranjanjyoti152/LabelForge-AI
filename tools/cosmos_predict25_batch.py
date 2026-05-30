#!/usr/bin/env python3
"""
Create and optionally run NVIDIA Cosmos Predict2.5 generation jobs.

This tool is intentionally host-side. Cosmos Predict2.5 is a large batch
generator, so Docker Compose runs it through the dedicated "cosmos" service.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_prompts(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Prompt file must contain a JSON list.")
    return data


def container_input_path(input_path: str) -> str:
    if input_path.startswith(("http://", "https://", "data:", "/workspace/", "/auto-labeler/")):
        return input_path
    return f"/auto-labeler/{input_path.lstrip('./')}"


def expand_jobs(prompts: List[Dict[str, Any]], jobs_dir: Path, default_inference_type: str) -> List[Path]:
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_paths: List[Path] = []

    for prompt_item in prompts:
        class_name = prompt_item.get("class_name")
        prompt = prompt_item.get("prompt")
        if not class_name or not prompt:
            raise ValueError("Each prompt item requires class_name and prompt.")

        count = int(prompt_item.get("count", 1))
        base_name = prompt_item.get("name") or str(class_name).replace(" ", "_")
        inference_type = prompt_item.get("inference_type", default_inference_type)

        for index in range(count):
            name = f"{base_name}_{index + 1:03d}" if count > 1 else base_name
            job: Dict[str, Any] = {
                "inference_type": inference_type,
                "name": name,
                "prompt": prompt,
                "class_name": class_name,
            }

            if prompt_item.get("negative_prompt"):
                job["negative_prompt"] = prompt_item["negative_prompt"]
            if prompt_item.get("input_path"):
                job["input_path"] = container_input_path(prompt_item["input_path"])
            if prompt_item.get("seed") is not None:
                job["seed"] = int(prompt_item["seed"]) + index
            for key in [
                "num_output_frames",
                "num_steps",
                "guidance",
                "resolution",
                "enable_autoregressive",
                "chunk_size",
                "chunk_overlap",
            ]:
                if key in prompt_item:
                    job[key] = prompt_item[key]

            job_path = jobs_dir / f"{name}.json"
            with open(job_path, "w") as f:
                json.dump(job, f, indent=2)
                f.write("\n")
            job_paths.append(job_path)

    return job_paths


def run_command(cmd: List[str], cwd: Path) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def cosmos_compose_cmd(compose_file: Path) -> List[str]:
    return ["docker", "compose", "-f", str(compose_file)]


def cosmos_service_name() -> str:
    return os.environ.get("COSMOS_COMPOSE_SERVICE", "cosmos")


def build_cosmos(compose_file: Path) -> None:
    run_command(cosmos_compose_cmd(compose_file) + ["build", cosmos_service_name()], REPO_ROOT)


def run_cosmos_job(compose_file: Path, job_path: Path, output_dir: Path, model: str) -> None:
    container_job = f"/auto-labeler/{job_path.relative_to(REPO_ROOT)}"
    container_output = f"/auto-labeler/{output_dir.relative_to(REPO_ROOT) / job_path.stem}"

    with open(job_path, "r") as f:
        inference_type = json.load(f).get("inference_type", "text2world")

    cmd = cosmos_compose_cmd(compose_file) + [
        "run",
        "--rm",
        cosmos_service_name(),
        "python",
        "examples/inference.py",
        "-i",
        container_job,
        "-o",
        container_output,
        "--inference-type",
        inference_type,
        "--model",
        model,
    ]
    run_command(cmd, REPO_ROOT)


def extract_frames(output_dir: Path, frames_dir: Path, frames_per_video: int) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(output_dir.rglob("*.mp4"))
    if not videos:
        print(f"No mp4 files found under {output_dir}")
        return

    for video in videos:
        frame_pattern = frames_dir / f"{video.stem}_%04d.jpg"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vf",
            "fps=2",
            "-frames:v",
            str(frames_per_video),
            str(frame_pattern),
        ]
        run_command(cmd, REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Cosmos Predict2.5 jobs for dataset balancing.")
    parser.add_argument("--prompts", default="prompts/cosmos_class_prompts.example.json",
                        help="JSON prompt file with class_name/prompt entries.")
    parser.add_argument("--jobs-dir", default="cosmos/jobs", help="Where to write Cosmos job JSON files.")
    parser.add_argument("--output-dir", default="cosmos/outputs", help="Where Cosmos writes generated videos.")
    parser.add_argument("--frames-dir", default="cosmos/frames", help="Where extracted frames are written.")
    parser.add_argument("--compose-file", default="docker-compose.yml", help="Compose file with the cosmos service.")
    parser.add_argument("--model", default=os.environ.get("COSMOS_MODEL", "2B/distilled"),
                        help="Cosmos model argument passed to examples/inference.py.")
    parser.add_argument("--default-inference-type",
                        default=os.environ.get("COSMOS_DEFAULT_INFERENCE_TYPE", "text2world"))
    parser.add_argument("--frames-per-video", type=int,
                        default=int(os.environ.get("COSMOS_FRAMES_PER_VIDEO", "12")))
    parser.add_argument("--build", action="store_true", help="Build the Cosmos Docker image.")
    parser.add_argument("--run", action="store_true", help="Run generated jobs in Docker.")
    parser.add_argument("--extract-frames", action="store_true", help="Extract frames from generated videos.")
    args = parser.parse_args()

    prompts_path = (REPO_ROOT / args.prompts).resolve()
    jobs_dir = (REPO_ROOT / args.jobs_dir).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    frames_dir = (REPO_ROOT / args.frames_dir).resolve()
    compose_file = (REPO_ROOT / args.compose_file).resolve()

    prompts = load_prompts(prompts_path)
    job_paths = expand_jobs(prompts, jobs_dir, args.default_inference_type)

    print(f"Created {len(job_paths)} Cosmos job file(s) in {jobs_dir}")

    if args.build:
        build_cosmos(compose_file)

    if args.run:
        for job_path in job_paths:
            run_cosmos_job(compose_file, job_path, output_dir, args.model)

    if args.extract_frames:
        extract_frames(output_dir, frames_dir, args.frames_per_video)

    if not args.build and not args.run and not args.extract_frames:
        print("Plan only. Add --build, --run, and/or --extract-frames to execute.")
        print("Next:")
        print(f"  python {Path(__file__).relative_to(REPO_ROOT)} --prompts {args.prompts} --build --run --extract-frames")

    return 0


if __name__ == "__main__":
    sys.exit(main())
