#!/usr/bin/env python3
"""
YOLO Dataset Preparation Tool

This tool fetches annotated tasks from Label Studio and converts them to YOLO format.
It can also auto-label unannotated images using the SAM3 detection server.

Features:
- Fetches tasks from Label Studio project
- Converts RectangleLabels to YOLO format
- Auto-labels images using SAM3 (optional)
- Splits dataset into train/val/test
- Creates dataset.yaml for YOLO training
- Progress bars and colored output
- Statistics and class distribution

Usage:
    python tools/prepare_yolo_dataset.py -p PROJECT_ID -o OUTPUT_DIR [options]

Examples:
    # Use existing annotations only
    python tools/prepare_yolo_dataset.py -p 1 -o ./datasets/traffic --use-existing

    # Auto-label all images with SAM3
    python tools/prepare_yolo_dataset.py -p 1 -o ./datasets/traffic --auto-label

    # Custom SAM3 server URL
    python tools/prepare_yolo_dataset.py -p 1 -o ./datasets/traffic --auto-label --sam3-url http://localhost:8080
"""

import argparse
import json
import math
import os
import sys
import shutil
import random
import time
import base64
import io
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def _strip_inline_comment(value: str) -> str:
    """Strip unquoted inline comments from a dotenv value."""
    quote = None
    escaped = False
    for i, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in ("'", '"'):
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            continue
        if char == "#" and quote is None and (i == 0 or value[i - 1].isspace()):
            return value[:i].rstrip()
    return value.strip()


def _load_env_file() -> None:
    """Load .env values even when python-dotenv is not installed."""
    env_path = Path(__file__).resolve().parent.parent / ".env"

    try:
        from dotenv import load_dotenv
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()  # Try current directory
        return
    except ImportError:
        pass

    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        value = _strip_inline_comment(value.strip())
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


# Load .env file automatically before reading defaults below.
_load_env_file()

import requests


# Default values
DEFAULT_LABEL_STUDIO_URL = os.environ.get("SAM3_LABELSTUDIO_API_BASE", "http://localhost:8080")
DEFAULT_SAM3_SERVER_URL = os.environ.get("SAM3_SERVER_URL", "http://localhost:8000")
DEFAULT_API_TOKEN = os.environ.get("SAM3_LABELSTUDIO_API_TOKEN", "")

# SAM3 detection configuration from .env
SAM3_SCORE_THRESHOLD = float(os.environ.get("SAM3_SCORE_THRESHOLD", "0.5"))
SAM3_NMS_THRESHOLD = float(os.environ.get("SAM3_NMS_THRESHOLD", "0.3"))
SAM3_MIN_BOX_AREA = int(os.environ.get("SAM3_MIN_BOX_AREA", "500"))
SAM3_MAX_DETECTIONS = int(os.environ.get("SAM3_MAX_DETECTIONS", "25"))
SAM3_CROSS_CLASS_NMS = os.environ.get("SAM3_CROSS_CLASS_NMS", "true").lower() == "true"

def cpu_worker_count() -> int:
    """Return available CPU cores, respecting Linux CPU affinity when present."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except Exception:
            pass
    return max(1, os.cpu_count() or 1)


def parse_worker_count(value: str, default: int) -> int:
    """Parse worker count values, allowing auto/max/all/0 for CPU-core count."""
    value = str(value).strip().lower()
    if value in ("auto", "max", "all", "0"):
        return cpu_worker_count()
    try:
        return max(1, int(value))
    except ValueError:
        return default


def gpu_memory_mb() -> Tuple[Optional[int], Optional[int]]:
    """Return (free_mb, total_mb) for the largest visible NVIDIA GPU."""
    if shutil.which("nvidia-smi") is None:
        return None, None

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None, None

    best: Tuple[Optional[int], Optional[int]] = (None, None)
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            free_mb = int(parts[0])
            total_mb = int(parts[1])
        except ValueError:
            continue
        if best[1] is None or total_mb > best[1]:
            best = (free_mb, total_mb)
    return best


def sam3_batch_size_for_vram(total_mb: Optional[int], free_mb: Optional[int]) -> int:
    """Choose a conservative SAM3 request batch size from detected VRAM."""
    if total_mb is None:
        return 4

    if total_mb < 8_000:
        batch_size = 2
    elif total_mb < 12_000:
        batch_size = 4
    elif total_mb < 16_000:
        batch_size = 6
    elif total_mb < 24_000:
        batch_size = 8
    elif total_mb < 32_000:
        batch_size = 12
    elif total_mb < 48_000:
        batch_size = 16
    elif total_mb < 80_000:
        batch_size = 24
    else:
        batch_size = 32

    if free_mb is not None:
        if free_mb < 4_000:
            batch_size = 1
        elif free_mb < 8_000:
            batch_size = min(batch_size, 4)
        elif free_mb < 12_000:
            batch_size = min(batch_size, 8)

    return max(1, batch_size)


def parse_sam3_batch_size(value: str | int, default: int = 4) -> int:
    """Parse SAM3 batch size, allowing auto/max/all/0 to size from NVIDIA VRAM."""
    value = str(value).strip().lower()
    if value in ("auto", "max", "all", "0"):
        free_mb, total_mb = gpu_memory_mb()
        return sam3_batch_size_for_vram(total_mb, free_mb)
    try:
        return max(1, int(value))
    except ValueError:
        return default


# Batch processing configuration
DEFAULT_BATCH_SIZE = int(os.environ.get("YOLO_BATCH_SIZE", "10"))
DEFAULT_WORKERS = parse_worker_count(os.environ.get("YOLO_WORKERS", "4"), 4)
DEFAULT_SAM3_BATCH_SIZE_RAW = os.environ.get("YOLO_SAM3_BATCH_SIZE", "auto")
DEFAULT_SAM3_BATCH_SIZE = parse_sam3_batch_size(DEFAULT_SAM3_BATCH_SIZE_RAW, 4)

# Dataset balancing / Cosmos Predict2.5 configuration
DEFAULT_MIN_SAMPLES_PER_CLASS = int(os.environ.get("DATASET_MIN_SAMPLES_PER_CLASS", "300"))
DEFAULT_COSMOS_SYNTHETIC_MAX_RATIO = float(os.environ.get("COSMOS_SYNTHETIC_MAX_RATIO", "0.35"))
DEFAULT_COSMOS_MODEL = os.environ.get("COSMOS_MODEL", "2B/distilled")
DEFAULT_COSMOS_FRAMES_PER_VIDEO = int(os.environ.get("COSMOS_FRAMES_PER_VIDEO", "12"))
DEFAULT_LM_STUDIO_API_BASE = os.environ.get(
    "LM_STUDIO_API_BASE",
    os.environ.get("LMSTUDIO_API_BASE", "http://localhost:1234/v1"),
)
DEFAULT_LM_STUDIO_MODEL = os.environ.get("LM_STUDIO_MODEL", os.environ.get("LMSTUDIO_MODEL", ""))
DEFAULT_LM_STUDIO_API_KEY = os.environ.get("LM_STUDIO_API_KEY", os.environ.get("LMSTUDIO_API_KEY", "lm-studio"))
DEFAULT_LABEL_STUDIO_TIMEOUT = int(os.environ.get("LABEL_STUDIO_REQUEST_TIMEOUT", "30"))
DEFAULT_IMAGE_DOWNLOAD_TIMEOUT = int(os.environ.get("IMAGE_DOWNLOAD_TIMEOUT", "30"))


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_banner():
    """Print the tool banner."""
    print(f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗
║  {Colors.BOLD}YOLO Dataset Preparation Tool{Colors.ENDC}{Colors.CYAN}                               ║
║  Label Studio → YOLO Format Converter                         ║
║  With SAM3 Auto-Labeling Support                              ║
╚══════════════════════════════════════════════════════════════╝{Colors.ENDC}
""")


def print_section(title: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}{Colors.ENDC}")


def format_time(seconds: float) -> str:
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def format_size(bytes: int) -> str:
    """Format bytes into human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024
    return f"{bytes:.1f} TB"


class Stats:
    """Track processing statistics (thread-safe)."""
    
    def __init__(self):
        self.start_time = time.time()
        self.total_tasks = 0
        self.processed = 0
        self.success = 0
        self.skipped = 0
        self.failed = 0
        self.total_annotations = 0
        self.total_bytes_downloaded = 0
        self.class_counts: Dict[str, int] = {}
        self.class_image_counts: Dict[str, int] = {}
        self.skipped_reasons: Dict[str, int] = {}
        self.failed_reasons: Dict[str, int] = {}
        self._lock = Lock()

    def add_annotations(self, count: int, labels: List[str]):
        with self._lock:
            self.total_annotations += count
            for label in labels:
                self.class_counts[label] = self.class_counts.get(label, 0) + 1
            for label in set(labels):
                self.class_image_counts[label] = self.class_image_counts.get(label, 0) + 1
    
    def add_bytes(self, bytes_count: int):
        with self._lock:
            self.total_bytes_downloaded += bytes_count
    
    def increment_success(self):
        with self._lock:
            self.processed += 1
            self.success += 1
    
    def increment_skipped(self, reason: str = "skipped"):
        with self._lock:
            self.processed += 1
            self.skipped += 1
            self.skipped_reasons[reason] = self.skipped_reasons.get(reason, 0) + 1
    
    def increment_failed(self, reason: str = "failed"):
        with self._lock:
            self.processed += 1
            self.failed += 1
            self.failed_reasons[reason] = self.failed_reasons.get(reason, 0) + 1
    
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    def eta(self) -> str:
        with self._lock:
            processed = self.processed
        if processed == 0:
            return "calculating..."
        elapsed = self.elapsed()
        rate = processed / elapsed
        remaining = (self.total_tasks - processed) / rate
        return format_time(remaining)
    
    def print_progress(self, task_id: Any = None, batch_info: str = ""):
        """Print current progress."""
        with self._lock:
            processed = self.processed
            success = self.success
            skipped = self.skipped
            failed = self.failed
        
        pct = (processed / self.total_tasks * 100) if self.total_tasks > 0 else 0
        elapsed = format_time(self.elapsed())
        eta = self.eta()
        
        # Create progress bar
        bar_width = 30
        filled = int(bar_width * processed / self.total_tasks) if self.total_tasks > 0 else 0
        bar = '█' * filled + '░' * (bar_width - filled)
        
        status = f"\r{Colors.CYAN}[{bar}]{Colors.ENDC} {pct:5.1f}% | "
        if batch_info:
            status += f"{batch_info} | "
        elif task_id:
            status += f"Task: {task_id} | "
        status += f"{Colors.GREEN}✓{success}{Colors.ENDC} "
        status += f"{Colors.YELLOW}⊘{skipped}{Colors.ENDC} "
        status += f"{Colors.RED}✗{failed}{Colors.ENDC} | "
        status += f"⏱ {elapsed} | ETA: {eta}  "
        
        print(status, end='', flush=True)
    
    def print_summary(self):
        """Print final summary."""
        print_section("Processing Summary")
        
        elapsed = format_time(self.elapsed())
        rate = self.processed / self.elapsed() if self.elapsed() > 0 else 0
        
        print(f"""
  {Colors.BOLD}Tasks Processed:{Colors.ENDC}
    • Total:     {self.total_tasks}
    • Success:   {Colors.GREEN}{self.success}{Colors.ENDC}
    • Skipped:   {Colors.YELLOW}{self.skipped}{Colors.ENDC}
    • Failed:    {Colors.RED}{self.failed}{Colors.ENDC}
  
  {Colors.BOLD}Annotations:{Colors.ENDC}
    • Total boxes: {self.total_annotations}
    • Downloaded:  {format_size(self.total_bytes_downloaded)}
  
  {Colors.BOLD}Performance:{Colors.ENDC}
    • Total time:  {elapsed}
    • Rate:        {rate:.2f} tasks/sec
""")
        
        if self.class_counts:
            print(f"  {Colors.BOLD}Class Distribution:{Colors.ENDC}")
            sorted_classes = sorted(self.class_counts.items(), key=lambda x: -x[1])
            max_count = max(self.class_counts.values())
            bar_max_width = 30
            
            for label, count in sorted_classes:
                bar_width = int(bar_max_width * count / max_count)
                bar = '█' * bar_width
                print(f"    {label:20s} {Colors.CYAN}{bar}{Colors.ENDC} {count}")

        if self.skipped_reasons:
            print(f"\n  {Colors.BOLD}Skipped Reasons:{Colors.ENDC}")
            for reason, count in sorted(self.skipped_reasons.items(), key=lambda x: -x[1]):
                print(f"    {reason:24s} {Colors.YELLOW}{count}{Colors.ENDC}")

        if self.failed_reasons:
            print(f"\n  {Colors.BOLD}Failed Reasons:{Colors.ENDC}")
            for reason, count in sorted(self.failed_reasons.items(), key=lambda x: -x[1]):
                print(f"    {reason:24s} {Colors.RED}{count}{Colors.ENDC}")


class LabelStudioClient:
    """Client for Label Studio API."""

    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Token {api_token}"}
        # Create a session for connection pooling (faster for multiple requests)
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_project(self, project_id: int) -> Dict[str, Any]:
        """Get project details including label config."""
        url = f"{self.base_url}/api/projects/{project_id}"
        resp = self.session.get(url, timeout=DEFAULT_LABEL_STUDIO_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def get_task_count(self, project_id: int) -> int:
        """Get total number of tasks in project."""
        url = f"{self.base_url}/api/projects/{project_id}"
        resp = self.session.get(url, timeout=DEFAULT_LABEL_STUDIO_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("task_number", 0)

    def get_all_tasks(self, project_id: int, max_tasks: int = 0, show_progress: bool = True) -> List[Dict[str, Any]]:
        """Fetch ALL tasks at once with large page size for efficiency."""
        all_tasks = []
        page = 1
        page_size = 100  # Fetch 100 at a time
        
        # Get total count first for progress bar
        total_count = self.get_task_count(project_id)
        if max_tasks > 0:
            total_count = min(total_count, max_tasks)
        
        while True:
            url = f"{self.base_url}/api/projects/{project_id}/tasks"
            params = {"page": page, "page_size": page_size}
            resp = self.session.get(url, params=params, timeout=DEFAULT_LABEL_STUDIO_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            
            if isinstance(data, list):
                batch = data
            else:
                batch = data.get("tasks", data.get("results", []))
            
            if not batch:
                break
            
            all_tasks.extend(batch)
            
            # Show progress bar
            if show_progress:
                fetched = len(all_tasks)
                pct = (fetched / total_count * 100) if total_count > 0 else 100
                bar_width = 30
                filled = int(bar_width * fetched / total_count) if total_count > 0 else bar_width
                bar = '█' * filled + '░' * (bar_width - filled)
                print(f"\r  {Colors.CYAN}[{bar}]{Colors.ENDC} {pct:5.1f}% | Fetched {fetched}/{total_count} tasks", end='', flush=True)
            
            # Check max_tasks limit
            if max_tasks > 0 and len(all_tasks) >= max_tasks:
                all_tasks = all_tasks[:max_tasks]
                break
            
            # Check if there are more pages
            if isinstance(data, dict) and data.get("next"):
                page += 1
            elif len(batch) < page_size:
                break
            else:
                page += 1
        
        if show_progress:
            print()  # New line after progress bar
        
        return all_tasks

    def iter_tasks(self, project_id: int, page_size: int = 1):
        """Iterate over tasks one by one (generator) - legacy method."""
        page = 1
        while True:
            url = f"{self.base_url}/api/projects/{project_id}/tasks"
            params = {"page": page, "page_size": page_size}
            resp = self.session.get(url, params=params, timeout=DEFAULT_LABEL_STUDIO_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            
            if isinstance(data, list):
                batch = data
            else:
                batch = data.get("tasks", data.get("results", []))
            
            if not batch:
                break
            
            for task in batch:
                yield task
            
            # Check if there are more pages
            if isinstance(data, dict) and data.get("next"):
                page += 1
            elif len(batch) < page_size:
                break
            else:
                page += 1

    def iter_task_batches(self, project_id: int, batch_size: int = 10):
        """Iterate over tasks in batches (generator yielding lists)."""
        page = 1
        while True:
            url = f"{self.base_url}/api/projects/{project_id}/tasks"
            params = {"page": page, "page_size": batch_size}
            resp = self.session.get(url, params=params, timeout=DEFAULT_LABEL_STUDIO_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            
            if isinstance(data, list):
                batch = data
            else:
                batch = data.get("tasks", data.get("results", []))
            
            if not batch:
                break
            
            yield batch
            
            # Check if there are more pages
            if isinstance(data, dict) and data.get("next"):
                page += 1
            elif len(batch) < batch_size:
                break
            else:
                page += 1

    def get_tasks(self, project_id: int, page_size: int = 100) -> List[Dict[str, Any]]:
        """Get all tasks from a project (for backward compatibility)."""
        return list(self.iter_tasks(project_id, page_size))

    def get_image_url(self, task: Dict[str, Any]) -> Optional[str]:
        """Extract image URL from task data."""
        data = task.get("data", {})
        
        # Try common field names
        for field in ["image", "img", "photo", "file", "url"]:
            if field in data:
                url = data[field]
                if isinstance(url, str):
                    # Handle Label Studio local file URLs
                    if url.startswith("/data/"):
                        return f"{self.base_url}{url}"
                    return url
        
        return None


class SAM3Client:
    """Client for SAM3 detection server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        # Load SAM3 config from environment
        self.score_threshold = SAM3_SCORE_THRESHOLD
        self.nms_threshold = SAM3_NMS_THRESHOLD
        self.min_box_area = SAM3_MIN_BOX_AREA
        self.max_detections = SAM3_MAX_DETECTIONS
        self.cross_class_nms = SAM3_CROSS_CLASS_NMS

    def health_check(self) -> bool:
        """Check if SAM3 server is running."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def predict_with_base64(self, image_base64: str, concepts: List[str]) -> List[Dict[str, Any]]:
        """Get predictions from SAM3 server using base64 image."""
        # Format: data:image/jpeg;base64,<base64_data>
        if not image_base64.startswith("data:"):
            image_base64 = f"data:image/jpeg;base64,{image_base64}"
        
        payload = {
            "tasks": [{
                "id": 0,
                "data": {
                    "image": image_base64,
                    "concepts": concepts,
                    # Pass SAM3 config for better detection
                    "score_threshold": self.score_threshold,
                    "nms_threshold": self.nms_threshold,
                    "min_box_area": self.min_box_area,
                    "max_detections": self.max_detections,
                    "cross_class_nms": self.cross_class_nms,
                }
            }]
        }
        
        resp = requests.post(
            f"{self.base_url}/predict",
            json=payload,
            timeout=120,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        
        data = resp.json()
        
        # Handle different response formats
        if "results" in data:
            results = data["results"]
        elif isinstance(data, list):
            results = data
        else:
            results = [data]
        
        if results and "result" in results[0]:
            return results[0]["result"]
        
        return []

    def predict(self, image_url: str, concepts: List[str]) -> List[Dict[str, Any]]:
        """Get predictions from SAM3 server (legacy URL method)."""
        payload = {
            "tasks": [{
                "id": 0,
                "data": {
                    "image": image_url,
                    "concepts": concepts,
                    # Pass SAM3 config for better detection
                    "score_threshold": self.score_threshold,
                    "nms_threshold": self.nms_threshold,
                    "min_box_area": self.min_box_area,
                    "max_detections": self.max_detections,
                    "cross_class_nms": self.cross_class_nms,
                }
            }]
        }
        
        resp = requests.post(
            f"{self.base_url}/predict",
            json=payload,
            timeout=120,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        
        data = resp.json()
        
        # Handle different response formats
        if "results" in data:
            results = data["results"]
        elif isinstance(data, list):
            results = data
        else:
            results = [data]
        
        if results and "result" in results[0]:
            return results[0]["result"]
        
        return []

    def predict_batch(self, images_base64: List[Tuple[int, str]], concepts: List[str]) -> Dict[int, List[Dict[str, Any]]]:
        """
        Batch prediction for multiple images.
        
        Args:
            images_base64: List of (task_id, base64_image_data) tuples
            concepts: List of concept labels to detect
            
        Returns:
            Dict mapping task_id to list of detection results
        """
        # Build batch request
        tasks = []
        for task_id, img_b64 in images_base64:
            if not img_b64.startswith("data:"):
                img_b64 = f"data:image/jpeg;base64,{img_b64}"
            
            tasks.append({
                "id": task_id,
                "data": {
                    "image": img_b64,
                    "concepts": concepts,
                    "score_threshold": self.score_threshold,
                    "nms_threshold": self.nms_threshold,
                    "min_box_area": self.min_box_area,
                    "max_detections": self.max_detections,
                    "cross_class_nms": self.cross_class_nms,
                }
            })
        
        payload = {"tasks": tasks}
        
        resp = requests.post(
            f"{self.base_url}/predict",
            json=payload,
            timeout=300,  # Longer timeout for batch
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        
        data = resp.json()
        
        # Parse response into dict by task_id
        results_map = {}
        
        if "results" in data:
            results_list = data["results"]
        elif isinstance(data, list):
            results_list = data
        else:
            results_list = [data]
        
        # Match results back to task IDs
        for i, result in enumerate(results_list):
            if i < len(images_base64):
                task_id = images_base64[i][0]
                results_map[task_id] = result.get("result", [])
        
        return results_map


def parse_labels_from_config(label_config: str) -> List[str]:
    """Parse label names from Label Studio XML config."""
    import re
    
    labels = []
    
    # Find all Label tags with value attribute
    pattern = r'<Label[^>]+value="([^"]+)"'
    matches = re.findall(pattern, label_config)
    labels.extend(matches)
    
    # Also check for labels in Choices
    pattern = r'<Choice[^>]+value="([^"]+)"'
    matches = re.findall(pattern, label_config)
    labels.extend(matches)
    
    return list(dict.fromkeys(labels))  # Remove duplicates, preserve order


def get_annotations_from_task(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract rectangle annotations from task."""
    annotations = []
    
    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            if result.get("type") == "rectanglelabels":
                value = result.get("value", {})
                labels = value.get("rectanglelabels", [])
                
                if labels:
                    annotations.append({
                        "label": labels[0],
                        "x": value.get("x", 0),
                        "y": value.get("y", 0),
                        "width": value.get("width", 0),
                        "height": value.get("height", 0),
                    })
    
    return annotations


def convert_to_yolo_format(
    annotations: List[Dict[str, Any]],
    label_to_id: Dict[str, int],
) -> List[str]:
    """Convert annotations to YOLO format strings."""
    yolo_lines = []
    
    for ann in annotations:
        label = ann["label"]
        if label not in label_to_id:
            continue
        
        class_id = label_to_id[label]
        
        # Label Studio uses percentages (0-100), YOLO uses normalized (0-1)
        x_center = (ann["x"] + ann["width"] / 2) / 100
        y_center = (ann["y"] + ann["height"] / 2) / 100
        width = ann["width"] / 100
        height = ann["height"] / 100
        
        # Clamp values to valid range
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_lines


# Colors for drawing bounding boxes (BGR for different classes)
BBOX_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
]


def draw_preview_image(
    image_data: bytes,
    annotations: List[Dict[str, Any]],
    label_to_id: Dict[str, int],
) -> Optional[bytes]:
    """Draw bounding boxes on image and return as bytes."""
    if not PIL_AVAILABLE:
        return None
    
    try:
        # Load image
        img = Image.open(io.BytesIO(image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Try to load a font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        for ann in annotations:
            label = ann["label"]
            if label not in label_to_id:
                continue
            
            class_id = label_to_id[label]
            color = BBOX_COLORS[class_id % len(BBOX_COLORS)]
            
            # Convert percentage to pixels
            x1 = int(ann["x"] * width / 100)
            y1 = int(ann["y"] * height / 100)
            x2 = int((ann["x"] + ann["width"]) * width / 100)
            y2 = int((ann["y"] + ann["height"]) * height / 100)
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label background
            text = f"{label}"
            bbox = draw.textbbox((x1, y1 - 20), text, font=font)
            draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=color)
            draw.text((x1, y1 - 20), text, fill=(255, 255, 255), font=font)
        
        # Save to bytes
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=90)
        return output.getvalue()
    
    except Exception as e:
        print(f"\n  {Colors.YELLOW}Warning: Could not draw preview: {e}{Colors.ENDC}")
        return None


def download_image(url: str, output_path: Path, headers: Dict[str, str] = None, session: requests.Session = None) -> int:
    """Download image and return bytes downloaded."""
    try:
        requester = session if session else requests
        resp = requester.get(url, headers=headers, timeout=DEFAULT_IMAGE_DOWNLOAD_TIMEOUT, stream=True)
        resp.raise_for_status()
        
        total_bytes = 0
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                total_bytes += len(chunk)
        
        return total_bytes
    except Exception as e:
        print(f"\n  {Colors.RED}✗ Download failed: {e}{Colors.ENDC}")
        return 0


def download_image_to_memory(url: str, headers: Dict[str, str] = None, session: requests.Session = None) -> Optional[bytes]:
    """Download image to memory and return bytes."""
    try:
        requester = session if session else requests
        resp = requester.get(url, headers=headers, timeout=DEFAULT_IMAGE_DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None


def process_task_download(
    task: Dict[str, Any],
    ls_client: LabelStudioClient,
) -> Tuple[int, Optional[bytes], Optional[str], str]:
    """
    Download image for a task (thread-safe, uses connection pooling).
    Returns (task_id, image_data, filename, status) tuple.
    """
    task_id = task.get("id", 0)
    
    # Get image URL
    image_url = ls_client.get_image_url(task)
    if not image_url:
        return task_id, None, None, "no_image_url"
    
    # Skip invalid URLs
    if not image_url.startswith(("http://", "https://")):
        return task_id, None, None, "invalid_url"
    
    # Determine output filename
    parsed = urlparse(image_url)
    original_name = Path(parsed.path).stem
    if not original_name or original_name in ["image", "img", "file"]:
        original_name = f"task_{task_id}"
    
    # Add task_id to ensure uniqueness
    filename = f"{original_name}_{task_id}"
    
    # Determine image extension
    image_ext = Path(parsed.path).suffix or ".jpg"
    if image_ext.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        image_ext = ".jpg"
    
    filename_with_ext = f"{filename}{image_ext}"
    
    # Download image using session for connection pooling
    image_data = download_image_to_memory(image_url, session=ls_client.session)
    if image_data is None:
        return task_id, None, filename_with_ext, "download_failed"
    
    return task_id, image_data, filename_with_ext, "success"


def process_batch(
    tasks: List[Dict[str, Any]],
    ls_client: LabelStudioClient,
    sam3_client: Optional[SAM3Client],
    images_dir: Path,
    labels_dir: Path,
    label_to_id: Dict[str, int],
    use_existing: bool,
    auto_label: bool,
    stats: Stats,
    project_labels: List[str] = None,
    preview_dir: Optional[Path] = None,
    num_workers: int = 4,
    sam3_batch_size: int = DEFAULT_SAM3_BATCH_SIZE,
) -> None:
    """
    Process a batch of tasks with parallel downloads and batch SAM3 prediction.
    """
    # Step 1: Download all images in parallel
    downloaded = {}  # task_id -> (image_data, filename, task)
    task_map = {t.get("id", i): t for i, t in enumerate(tasks)}
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_task_download, task, ls_client): task.get("id", i)
            for i, task in enumerate(tasks)
        }
        
        for future in as_completed(futures):
            task_id, image_data, filename, status = future.result()
            
            if status == "success" and image_data is not None:
                downloaded[task_id] = (image_data, filename, task_map[task_id])
            else:
                stats.increment_skipped(status)
    
    if not downloaded:
        return
    
    # Step 2: Get existing annotations or prepare for auto-labeling
    annotations_map = {}  # task_id -> list of annotations
    need_auto_label = []  # list of (task_id, image_base64) for SAM3
    
    for task_id, (image_data, filename, task) in downloaded.items():
        annotations = []
        
        if use_existing:
            annotations = get_annotations_from_task(task)
        
        if annotations:
            annotations_map[task_id] = annotations
        elif auto_label and sam3_client:
            # Queue for batch auto-labeling
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            need_auto_label.append((task_id, image_base64))
        else:
            # No annotations and no auto-label - mark as skipped
            stats.increment_skipped("no_annotations")
    
    # Step 3: Batch SAM3 prediction for tasks needing auto-labeling
    if need_auto_label and sam3_client:
        concepts = project_labels or list(label_to_id.keys())
        for batch_start in range(0, len(need_auto_label), max(1, sam3_batch_size)):
            sam3_chunk = need_auto_label[batch_start:batch_start + max(1, sam3_batch_size)]
            try:
                sam3_results = sam3_client.predict_batch(sam3_chunk, concepts)
            except requests.exceptions.ConnectionError:
                for task_id, _ in sam3_chunk:
                    stats.increment_failed("sam3_connection_error")
                continue
            except requests.exceptions.Timeout:
                for task_id, _ in sam3_chunk:
                    stats.increment_failed("sam3_timeout")
                continue
            except Exception as e:
                reason = f"sam3_error:{type(e).__name__}"
                for task_id, _ in sam3_chunk:
                    stats.increment_failed(reason)
                continue
            
            for task_id, results in sam3_results.items():
                annotations = []
                for result in results:
                    value = result.get("value", {})
                    labels = value.get("rectanglelabels", [])
                    if labels:
                        annotations.append({
                            "label": labels[0],
                            "x": value.get("x", 0),
                            "y": value.get("y", 0),
                            "width": value.get("width", 0),
                            "height": value.get("height", 0),
                        })
                
                if annotations:
                    annotations_map[task_id] = annotations
                else:
                    stats.increment_skipped("sam3_no_detections")
    
    # Step 4: Save images and labels (can be done in parallel)
    def save_task(task_id: int) -> Tuple[bool, str]:
        if task_id not in downloaded or task_id not in annotations_map:
            return False, "missing_data"
        
        image_data, filename, task = downloaded[task_id]
        annotations = annotations_map[task_id]
        
        # Convert to YOLO format
        yolo_lines = convert_to_yolo_format(annotations, label_to_id)
        if not yolo_lines:
            return False, "no_valid_boxes"
        
        # Save image
        image_path = images_dir / filename
        with open(image_path, "wb") as f:
            f.write(image_data)
        
        # Save labels
        stem = Path(filename).stem
        label_path = labels_dir / f"{stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))
        
        # Save preview image with bounding boxes
        if preview_dir is not None:
            preview_data = draw_preview_image(image_data, annotations, label_to_id)
            if preview_data:
                preview_path = preview_dir / f"{stem}_preview.jpg"
                with open(preview_path, "wb") as f:
                    f.write(preview_data)
        
        # Update stats
        stats.add_bytes(len(image_data))
        labels_in_task = [ann["label"] for ann in annotations if ann["label"] in label_to_id]
        stats.add_annotations(len(yolo_lines), labels_in_task)
        
        return True, "success"
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(save_task, task_id): task_id
            for task_id in annotations_map.keys()
        }
        
        for future in as_completed(futures):
            success, reason = future.result()
            if success:
                stats.increment_success()
            else:
                stats.increment_skipped(reason)


def process_task(
    task: Dict[str, Any],
    ls_client: LabelStudioClient,
    sam3_client: Optional[SAM3Client],
    images_dir: Path,
    labels_dir: Path,
    label_to_id: Dict[str, int],
    use_existing: bool,
    auto_label: bool,
    stats: Stats,
    project_labels: List[str] = None,
    preview_dir: Optional[Path] = None,
) -> Tuple[bool, str]:
    """
    Process a single task and save image + labels.
    Returns (success, reason) tuple.
    """
    task_id = task.get("id", "unknown")
    
    # Get image URL
    image_url = ls_client.get_image_url(task)
    if not image_url:
        return False, "no_image_url"
    
    # Determine output filename early
    parsed = urlparse(image_url)
    original_name = Path(parsed.path).stem
    if not original_name or original_name in ["image", "img", "file"]:
        original_name = f"task_{task_id}"
    
    # Add task_id to ensure uniqueness
    filename = f"{original_name}_{task_id}"
    
    # Determine image extension
    image_ext = Path(parsed.path).suffix or ".jpg"
    if image_ext.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        image_ext = ".jpg"
    
    image_path = images_dir / f"{filename}{image_ext}"
    
    # Download image first (needed for both auto-label and saving)
    image_data = None
    try:
        # Skip invalid URLs
        if not image_url.startswith(("http://", "https://")):
            return False, "invalid_url"
        resp = requests.get(image_url, headers=ls_client.headers, timeout=DEFAULT_IMAGE_DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
        image_data = resp.content
        stats.add_bytes(len(image_data))
    except Exception as e:
        return False, "download_failed"
    
    # Get annotations
    annotations = []
    
    if use_existing:
        annotations = get_annotations_from_task(task)
    
    if not annotations and auto_label and sam3_client:
        # Use project labels for auto-labeling
        concepts = project_labels or list(label_to_id.keys())
        try:
            # Convert image to base64 and send to SAM3
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            sam3_results = sam3_client.predict_with_base64(image_base64, concepts)
            for result in sam3_results:
                value = result.get("value", {})
                labels = value.get("rectanglelabels", [])
                if labels:
                    annotations.append({
                        "label": labels[0],
                        "x": value.get("x", 0),
                        "y": value.get("y", 0),
                        "width": value.get("width", 0),
                        "height": value.get("height", 0),
                    })
        except requests.exceptions.ConnectionError:
            return False, "sam3_connection_error"
        except requests.exceptions.Timeout:
            return False, "sam3_timeout"
        except Exception as e:
            return False, f"sam3_error:{str(e)[:50]}"
    
    if not annotations:
        return False, "no_annotations"
    
    # Convert to YOLO format
    yolo_lines = convert_to_yolo_format(annotations, label_to_id)
    if not yolo_lines:
        return False, "no_valid_boxes"
    
    # Save image
    with open(image_path, "wb") as f:
        f.write(image_data)
    
    # Save labels
    label_path = labels_dir / f"{Path(filename).stem}.txt"
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines))
    
    # Save preview image with bounding boxes
    if preview_dir is not None:
        preview_data = draw_preview_image(image_data, annotations, label_to_id)
        if preview_data:
            preview_path = preview_dir / f"{filename}_preview.jpg"
            with open(preview_path, "wb") as f:
                f.write(preview_data)
    
    # Update stats
    labels_in_task = [ann["label"] for ann in annotations if ann["label"] in label_to_id]
    stats.add_annotations(len(yolo_lines), labels_in_task)
    
    return True, "success"


def split_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    train_split: float,
    val_split: float,
    test_split: float,
) -> Dict[str, int]:
    """Split dataset into train/val/test sets."""
    
    # Get all image files
    image_files = list(images_dir.glob("*"))
    image_files = [f for f in image_files if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
    
    random.shuffle(image_files)
    
    n = len(image_files)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    # Create output directories
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    def copy_files(files: List[Path], split: str):
        for img_file in files:
            # Copy image
            dst_img = output_dir / split / "images" / img_file.name
            shutil.copy2(img_file, dst_img)
            
            # Copy label
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                dst_label = output_dir / split / "labels" / label_file.name
                shutil.copy2(label_file, dst_label)
    
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")
    
    return {
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
    }


def create_dataset_yaml(output_dir: Path, labels: List[str], project_name: str):
    """Create dataset.yaml for YOLO training."""
    yaml_content = f"""# Dataset config for {project_name}
# Generated by prepare_yolo_dataset.py

path: {output_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
names:
"""
    for i, label in enumerate(labels):
        yaml_content += f"  {i}: {label}\n"
    
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    print(f"{Colors.GREEN}✓ Created dataset.yaml at {yaml_path}{Colors.ENDC}")


def validate_output_dir(output_dir: Path, force: bool = False) -> None:
    """Fail fast when the dataset output path cannot be created or replaced."""
    if output_dir.exists():
        if not force:
            print(f"{Colors.RED}✗ Output directory exists. Use --force to overwrite{Colors.ENDC}")
            sys.exit(1)
        if not os.access(output_dir, os.W_OK | os.X_OK):
            print(f"{Colors.RED}✗ Output directory is not writable: {output_dir}{Colors.ENDC}")
            print("  Fix ownership/permissions or choose a different --output-dir.")
            sys.exit(1)
        return

    parent = output_dir.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent

    if not parent.exists() or not os.access(parent, os.W_OK | os.X_OK):
        print(f"{Colors.RED}✗ Cannot create output directory: {output_dir}{Colors.ENDC}")
        print(f"  Parent directory is not writable: {parent}")
        print("  Fix ownership/permissions or choose a different --output-dir.")
        sys.exit(1)


def slugify_name(value: str) -> str:
    """Create a filesystem-safe slug while keeping class names readable."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "class"


def project_path(pattern: str, project_id: int) -> Path:
    """Resolve a path pattern that may include {project_id}."""
    path = Path(pattern.format(project_id=project_id))
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent.parent / path


def identify_weak_classes(
    labels: List[str],
    class_counts: Dict[str, int],
    min_samples_per_class: int,
    synthetic_max_ratio: float,
    frames_per_video: int,
) -> List[Dict[str, Any]]:
    """Find classes that need extra train-only synthetic examples."""
    weak_classes: List[Dict[str, Any]] = []
    frames_per_video = max(1, frames_per_video)

    for label in labels:
        real_count = int(class_counts.get(label, 0))
        if real_count >= min_samples_per_class:
            continue

        deficit = min_samples_per_class - real_count
        synthetic_cap_basis = max(real_count, min_samples_per_class)
        max_synthetic = max(1, int(synthetic_cap_basis * synthetic_max_ratio))
        target_synthetic = min(deficit, max_synthetic)
        video_count = max(1, math.ceil(target_synthetic / frames_per_video))

        weak_classes.append({
            "class_name": label,
            "current_count": real_count,
            "min_samples": min_samples_per_class,
            "deficit": deficit,
            "target_synthetic_frames": target_synthetic,
            "cosmos_video_count": video_count,
        })

    return weak_classes


class LMStudioPromptClient:
    """OpenAI-compatible LM Studio client for project-specific Cosmos prompts."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate_prompts(
        self,
        project_title: str,
        labels: List[str],
        weak_classes: List[Dict[str, Any]],
        frames_per_video: int,
    ) -> List[Dict[str, Any]]:
        if not self.model:
            raise ValueError("LM Studio model is not configured.")

        system_prompt = (
            "You create high-quality NVIDIA Cosmos text2world prompts for object detection dataset balancing. "
            "Return only valid JSON. Do not include markdown. Prompts must be realistic camera scenes, "
            "visually diverse, and useful for YOLO object detection. Avoid text overlays, watermarks, logos, "
            "distorted objects, extreme occlusion, and synthetic-looking studio renders."
        )
        user_payload = {
            "project_title": project_title,
            "all_project_classes": labels,
            "weak_classes": weak_classes,
            "frames_per_video": frames_per_video,
            "required_output_schema": {
                "prompts": [{
                    "class_name": "exact class from weak_classes",
                    "name": "short filesystem-safe job name",
                    "inference_type": "text2world",
                    "prompt": "single detailed prompt",
                    "negative_prompt": "single negative prompt",
                    "count": "cosmos_video_count for this class",
                    "seed": "integer seed",
                }]
            },
            "rules": [
                "Create exactly one prompt item per weak class.",
                "Use each weak class name exactly as provided.",
                "The target object should be clearly visible, realistic, and large enough for detection.",
                "Mention natural context, camera angle, lighting, background, and object variety.",
                "Do not create examples for classes that are not in weak_classes.",
                "Set count equal to cosmos_video_count.",
            ],
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, indent=2)},
            ],
            "temperature": 0.4,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = parse_lmstudio_json(content)
        prompts = parsed.get("prompts") if isinstance(parsed, dict) else parsed
        return normalize_cosmos_prompts(prompts, weak_classes)


def parse_lmstudio_json(content: str) -> Any:
    """Parse JSON from LM Studio, tolerating accidental code fences."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def normalize_cosmos_prompts(prompts: Any, weak_classes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only valid prompt fields and enforce counts/class names."""
    if not isinstance(prompts, list):
        raise ValueError("LM Studio response must include a prompts list.")

    weak_by_name = {item["class_name"]: item for item in weak_classes}
    normalized: List[Dict[str, Any]] = []
    for item in prompts:
        if not isinstance(item, dict):
            continue
        class_name = item.get("class_name")
        prompt = item.get("prompt")
        if class_name not in weak_by_name or not prompt:
            continue

        weak = weak_by_name[class_name]
        normalized.append({
            "class_name": class_name,
            "name": slugify_name(str(item.get("name") or class_name)),
            "inference_type": item.get("inference_type", "text2world"),
            "prompt": str(prompt).strip(),
            "negative_prompt": str(item.get("negative_prompt") or default_negative_prompt()).strip(),
            "count": int(weak["cosmos_video_count"]),
            "seed": int(item.get("seed", 1000 + len(normalized) * 100)),
        })

    missing = [item["class_name"] for item in weak_classes if item["class_name"] not in {p["class_name"] for p in normalized}]
    if missing:
        raise ValueError(f"LM Studio did not return valid prompts for: {', '.join(missing)}")
    return normalized


def default_negative_prompt() -> str:
    return (
        "text, watermark, logo, cartoon, blurry, duplicate objects merged together, distorted geometry, "
        "unrealistic scale, heavy occlusion, tiny target object, overexposed, underexposed"
    )


def fallback_cosmos_prompts(
    project_title: str,
    weak_classes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Deterministic fallback when LM Studio is unavailable."""
    prompts: List[Dict[str, Any]] = []
    for index, weak in enumerate(weak_classes):
        class_name = weak["class_name"]
        prompts.append({
            "class_name": class_name,
            "name": f"{slugify_name(class_name)}_balanced",
            "inference_type": "text2world",
            "prompt": (
                f"Realistic video from the {project_title} dataset context showing one or more clearly visible "
                f"{class_name} objects in natural surroundings. Use varied camera distance, practical lighting, "
                f"real-world background clutter, and enough object size for accurate object detection labels."
            ),
            "negative_prompt": default_negative_prompt(),
            "count": int(weak["cosmos_video_count"]),
            "seed": 1000 + index * 100,
        })
    return prompts


def write_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def run_cosmos_prompt_batch(
    prompts_path: Path,
    jobs_dir: Path,
    cosmos_output_dir: Path,
    frames_dir: Path,
    model: str,
    frames_per_video: int,
    build: bool,
) -> None:
    """Create Cosmos jobs, run Docker generation, and extract frames."""
    script_path = Path(__file__).resolve().parent / "cosmos_predict25_batch.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--prompts",
        str(prompts_path),
        "--jobs-dir",
        str(jobs_dir),
        "--output-dir",
        str(cosmos_output_dir),
        "--frames-dir",
        str(frames_dir),
        "--model",
        model,
        "--frames-per-video",
        str(frames_per_video),
        "--run",
        "--extract-frames",
    ]
    if build:
        cmd.append("--build")
    subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent, check=True)


def load_job_class_map(jobs_dir: Path) -> Dict[str, str]:
    """Map generated Cosmos job stems to their target class."""
    job_map: Dict[str, str] = {}
    if not jobs_dir.exists():
        return job_map

    class_map_path = jobs_dir / "_class_map.json"
    if class_map_path.exists():
        try:
            with open(class_map_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                job_map.update({str(k): str(v) for k, v in data.items()})
        except Exception:
            pass

    for job_path in sorted(jobs_dir.glob("*.json")):
        if job_path.name == "_class_map.json":
            continue
        try:
            with open(job_path, "r") as f:
                data = json.load(f)
            if data.get("class_name"):
                job_map[job_path.stem] = data["class_name"]
        except Exception:
            continue
    return job_map


def class_for_frame(frame_path: Path, job_class_map: Dict[str, str]) -> Optional[str]:
    """Infer target class from an extracted frame filename."""
    frame_stem = frame_path.stem
    for job_stem, class_name in sorted(job_class_map.items(), key=lambda item: len(item[0]), reverse=True):
        if frame_stem == job_stem or frame_stem.startswith(f"{job_stem}_"):
            return class_name
    return None


def label_synthetic_frames(
    frames_dir: Path,
    jobs_dir: Path,
    output_dir: Path,
    label_to_id: Dict[str, int],
    sam3_client: SAM3Client,
    preview_dir: Optional[Path],
) -> Dict[str, Any]:
    """Run SAM3 over Cosmos frames and add accepted labels to train split."""
    train_images_dir = output_dir / "train" / "images"
    train_labels_dir = output_dir / "train" / "labels"
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)

    job_class_map = load_job_class_map(jobs_dir)
    image_files = [
        path for path in sorted(frames_dir.rglob("*"))
        if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ]

    report: Dict[str, Any] = {
        "frames_seen": len(image_files),
        "accepted_images": 0,
        "rejected_images": 0,
        "accepted_boxes": 0,
        "accepted_by_class": {},
        "rejected_reasons": {},
    }

    for frame_path in image_files:
        class_name = class_for_frame(frame_path, job_class_map)
        if class_name not in label_to_id:
            report["rejected_images"] += 1
            report["rejected_reasons"]["unknown_class"] = report["rejected_reasons"].get("unknown_class", 0) + 1
            continue

        try:
            image_data = frame_path.read_bytes()
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            annotations = sam3_client.predict_with_base64(image_base64, [class_name])
            parsed_annotations = []
            for result in annotations:
                value = result.get("value", {})
                labels = value.get("rectanglelabels", [])
                if labels and labels[0] == class_name:
                    parsed_annotations.append({
                        "label": labels[0],
                        "x": value.get("x", 0),
                        "y": value.get("y", 0),
                        "width": value.get("width", 0),
                        "height": value.get("height", 0),
                    })
            annotations = parsed_annotations
            yolo_lines = convert_to_yolo_format(annotations, label_to_id)
        except Exception as e:
            report["rejected_images"] += 1
            reason = f"sam3_error:{str(e)[:50]}"
            report["rejected_reasons"][reason] = report["rejected_reasons"].get(reason, 0) + 1
            continue

        if not yolo_lines:
            report["rejected_images"] += 1
            report["rejected_reasons"]["no_valid_boxes"] = report["rejected_reasons"].get("no_valid_boxes", 0) + 1
            continue

        class_slug = slugify_name(class_name)
        dst_stem = f"cosmos_{class_slug}_{report['accepted_images'] + 1:06d}"
        dst_image = train_images_dir / f"{dst_stem}.jpg"
        dst_label = train_labels_dir / f"{dst_stem}.txt"

        shutil.copy2(frame_path, dst_image)
        with open(dst_label, "w") as f:
            f.write("\n".join(yolo_lines))

        if preview_dir is not None:
            preview_data = draw_preview_image(image_data, annotations, label_to_id)
            if preview_data:
                with open(preview_dir / f"{dst_stem}_preview.jpg", "wb") as f:
                    f.write(preview_data)

        report["accepted_images"] += 1
        report["accepted_boxes"] += len(yolo_lines)
        accepted_by_class = report["accepted_by_class"]
        accepted_by_class[class_name] = accepted_by_class.get(class_name, 0) + len(yolo_lines)

    return report


def write_quality_report(
    output_dir: Path,
    project: Dict[str, Any],
    labels: List[str],
    real_counts: Dict[str, int],
    skipped_reasons: Dict[str, int],
    failed_reasons: Dict[str, int],
    weak_classes: List[Dict[str, Any]],
    prompts_path: Optional[Path],
    synthetic_report: Optional[Dict[str, Any]],
) -> None:
    """Write a compact machine-readable dataset quality report."""
    report = {
        "project_id": project.get("id"),
        "project_title": project.get("title", "Unknown"),
        "labels": labels,
        "real_class_counts": {label: int(real_counts.get(label, 0)) for label in labels},
        "skipped_reasons": skipped_reasons,
        "failed_reasons": failed_reasons,
        "weak_classes": weak_classes,
        "cosmos_prompts": str(prompts_path) if prompts_path else None,
        "synthetic": synthetic_report,
    }
    write_json_file(output_dir / "quality_report.json", report)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare YOLO dataset from Label Studio annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use existing annotations only
  python tools/prepare_yolo_dataset.py -p 1 -o ./datasets/traffic --use-existing

  # Auto-label all images with SAM3
  python tools/prepare_yolo_dataset.py -p 1 -o ./datasets/traffic --auto-label

  # Both: use existing, auto-label the rest
  python tools/prepare_yolo_dataset.py -p 1 -o ./datasets/traffic --use-existing --auto-label
        """,
    )
    
    parser.add_argument("-p", "--project-id", type=int, required=True,
                        help="Label Studio project ID")
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="Output directory for YOLO dataset")
    parser.add_argument("--ls-url", type=str, default=DEFAULT_LABEL_STUDIO_URL,
                        help=f"Label Studio URL (default: {DEFAULT_LABEL_STUDIO_URL})")
    parser.add_argument("--ls-token", type=str, default=DEFAULT_API_TOKEN,
                        help="Label Studio API token")
    parser.add_argument("--sam3-url", type=str, default=DEFAULT_SAM3_SERVER_URL,
                        help=f"SAM3 server URL (default: {DEFAULT_SAM3_SERVER_URL})")
    parser.add_argument("--use-existing", action="store_true",
                        help="Use existing annotations from Label Studio")
    parser.add_argument("--auto-label", action="store_true",
                        help="Auto-label images using SAM3")
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Train split ratio (default: 0.8)")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Validation split ratio (default: 0.15)")
    parser.add_argument("--test-split", type=float, default=0.05,
                        help="Test split ratio (default: 0.05)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output directory")
    parser.add_argument("--save-preview", action="store_true",
                        help="Save labeled preview images to ~/labeled_previews")
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Maximum number of tasks to process (0 = all)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Number of tasks to download/process per batch (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--sam3-batch-size", type=str, default=str(DEFAULT_SAM3_BATCH_SIZE),
                        help=f"Images per SAM3 /predict request; use auto/0 to size from VRAM (default: {DEFAULT_SAM3_BATCH_SIZE})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of parallel workers for downloads; use 0 for CPU-core count (default: {DEFAULT_WORKERS})")
    parser.add_argument("--max-workers", action="store_true",
                        help="Use all available CPU cores for download/processing workers")
    parser.add_argument("--no-batch", action="store_true",
                        help="Disable batch processing (process one-by-one)")
    parser.add_argument("--preload-tasks", action="store_true",
                        help="Fetch all Label Studio task metadata before processing (old behavior)")
    parser.add_argument("--balance-classes", action="store_true",
                        help="Report classes below --min-samples-per-class")
    parser.add_argument("--cosmos-augment", action="store_true",
                        help="Use LM Studio + Cosmos Predict2.5 + SAM3 to add train-only samples for weak classes")
    parser.add_argument("--min-samples-per-class", type=int, default=DEFAULT_MIN_SAMPLES_PER_CLASS,
                        help=f"Minimum desired boxes per class (default: {DEFAULT_MIN_SAMPLES_PER_CLASS})")
    parser.add_argument("--synthetic-max-ratio", type=float, default=DEFAULT_COSMOS_SYNTHETIC_MAX_RATIO,
                        help=f"Maximum synthetic frames as a ratio of class target basis (default: {DEFAULT_COSMOS_SYNTHETIC_MAX_RATIO})")
    parser.add_argument("--cosmos-model", type=str, default=DEFAULT_COSMOS_MODEL,
                        help=f"Cosmos model passed to examples/inference.py (default: {DEFAULT_COSMOS_MODEL})")
    parser.add_argument("--cosmos-build", action="store_true",
                        help="Build the Cosmos Predict2.5 Docker image before generation")
    parser.add_argument("--cosmos-frames-per-video", type=int, default=DEFAULT_COSMOS_FRAMES_PER_VIDEO,
                        help=f"Frames to extract per generated video (default: {DEFAULT_COSMOS_FRAMES_PER_VIDEO})")
    parser.add_argument("--cosmos-prompts-out", type=str, default="cosmos/prompts/project_{project_id}_prompts.json",
                        help="Where generated Cosmos prompts are saved; supports {project_id}")
    parser.add_argument("--cosmos-jobs-dir", type=str, default="cosmos/jobs/project_{project_id}",
                        help="Where generated Cosmos job JSON files are saved; supports {project_id}")
    parser.add_argument("--cosmos-output-dir", type=str, default="cosmos/outputs/project_{project_id}",
                        help="Where Cosmos writes generated videos; supports {project_id}")
    parser.add_argument("--cosmos-frames-dir", type=str, default="cosmos/frames/project_{project_id}",
                        help="Where extracted Cosmos frames are saved; supports {project_id}")
    parser.add_argument("--lmstudio-url", type=str, default=DEFAULT_LM_STUDIO_API_BASE,
                        help=f"LM Studio OpenAI-compatible API base (default: {DEFAULT_LM_STUDIO_API_BASE})")
    parser.add_argument("--lmstudio-model", type=str, default=DEFAULT_LM_STUDIO_MODEL,
                        help="LM Studio model name used to generate Cosmos prompts")
    parser.add_argument("--lmstudio-api-key", type=str, default=DEFAULT_LM_STUDIO_API_KEY,
                        help="LM Studio API key if configured")
    parser.add_argument("--lmstudio-timeout", type=int, default=120,
                        help="LM Studio request timeout in seconds")
    
    args = parser.parse_args()
    args.sam3_batch_size = parse_sam3_batch_size(args.sam3_batch_size, DEFAULT_SAM3_BATCH_SIZE)

    if args.cosmos_augment:
        args.balance_classes = True
    if args.max_workers:
        args.workers = cpu_worker_count()
        args.batch_size = max(args.batch_size, args.workers)
    elif args.workers <= 0:
        args.workers = cpu_worker_count()

    if (args.auto_label or args.cosmos_augment) and not args.no_batch:
        args.batch_size = max(args.batch_size, args.sam3_batch_size)
    
    # Validate arguments
    if not args.use_existing and not args.auto_label:
        print(f"{Colors.YELLOW}⚠ Neither --use-existing nor --auto-label specified.")
        print(f"  Defaulting to --use-existing{Colors.ENDC}")
        args.use_existing = True

    total_split = args.train_split + args.val_split + args.test_split
    if abs(total_split - 1.0) > 0.001:
        print(f"{Colors.RED}✗ --train-split + --val-split + --test-split must equal 1.0, got {total_split:.4f}{Colors.ENDC}")
        sys.exit(1)
    
    if not args.ls_token:
        print(f"{Colors.RED}✗ Label Studio API token required. Set SAM3_LABELSTUDIO_API_TOKEN or use --ls-token{Colors.ENDC}")
        sys.exit(1)
    
    # Print banner
    print_banner()
    
    # Print configuration
    print_section("Configuration")
    print(f"""
  Project ID:      {args.project_id}
  Label Studio:    {args.ls_url}
  SAM3 Server:     {args.sam3_url}
  Output Dir:      {Path(args.output_dir).absolute()}
  Use Existing:    {'Yes' if args.use_existing else 'No'}
  Auto-Label:      {'Yes' if args.auto_label else 'No'}
  Balance Classes: {'Yes' if args.balance_classes else 'No'}
  Cosmos Augment:  {'Yes' if args.cosmos_augment else 'No'}
  Split Ratio:     train={args.train_split}, val={args.val_split}, test={args.test_split}
  CPU Cores:       {cpu_worker_count()}
  Task Loading:    {'Preload all tasks' if args.preload_tasks or args.no_batch else 'Stream batches'}
  Batch Mode:      {'Disabled' if args.no_batch else f'Yes (batch_size={args.batch_size}, sam3_batch_size={args.sam3_batch_size}, workers={args.workers})'}
""")
    
    if args.auto_label or args.cosmos_augment:
        free_mb, total_mb = gpu_memory_mb()
        print(f"  {Colors.BOLD}SAM3 Detection Config (from .env):{Colors.ENDC}")
        print(f"    • Score Threshold:  {SAM3_SCORE_THRESHOLD}")
        print(f"    • NMS Threshold:    {SAM3_NMS_THRESHOLD}")
        print(f"    • Min Box Area:     {SAM3_MIN_BOX_AREA}")
        print(f"    • Max Detections:   {SAM3_MAX_DETECTIONS}")
        print(f"    • Cross-Class NMS:  {SAM3_CROSS_CLASS_NMS}")
        if total_mb is not None:
            print(f"    • GPU VRAM:         {free_mb}MB free / {total_mb}MB total")
        print(f"    • SAM3 Batch Size:  {args.sam3_batch_size}")
        print()

    if args.cosmos_augment:
        print(f"  {Colors.BOLD}Cosmos / LM Studio Config:{Colors.ENDC}")
        print(f"    • Min Samples/Class: {args.min_samples_per_class}")
        print(f"    • Synthetic Ratio:   {args.synthetic_max_ratio}")
        print(f"    • Cosmos Model:      {args.cosmos_model}")
        print(f"    • Frames/Video:      {args.cosmos_frames_per_video}")
        print(f"    • LM Studio URL:     {args.lmstudio_url}")
        print(f"    • LM Studio Model:   {args.lmstudio_model or '(not set; fallback prompts)'}")
        print()

    output_dir = Path(args.output_dir)
    validate_output_dir(output_dir, force=args.force)
    
    # Initialize clients
    ls_client = LabelStudioClient(args.ls_url, args.ls_token)
    sam3_client = None
    
    if args.auto_label or args.cosmos_augment:
        sam3_client = SAM3Client(args.sam3_url)
        if not sam3_client.health_check():
            print(f"{Colors.RED}✗ SAM3 server not responding at {args.sam3_url}{Colors.ENDC}")
            print(f"  Start the server with: uvicorn app.main:app --port 8080")
            sys.exit(1)
        print(f"{Colors.GREEN}✓ SAM3 server connected{Colors.ENDC}")
    
    # Get project info
    print_section("Fetching Project Info")
    try:
        project = ls_client.get_project(args.project_id)
        print(f"{Colors.GREEN}✓ Connected to project: {project.get('title', 'Unknown')}{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.RED}✗ Failed to get project: {e}{Colors.ENDC}")
        sys.exit(1)
    
    # Parse labels from config
    label_config = project.get("label_config", "")
    labels = parse_labels_from_config(label_config)
    
    if not labels:
        print(f"{Colors.RED}✗ No labels found in project config{Colors.ENDC}")
        sys.exit(1)
    
    print(f"{Colors.GREEN}✓ Found {len(labels)} labels (will use these for auto-labeling):{Colors.ENDC}")
    for i, label in enumerate(labels):
        print(f"    {i}: {label}")
    
    label_to_id = {label: i for i, label in enumerate(labels)}
    
    # Fetch task count, then either stream task batches or preload task metadata.
    print_section("Fetching Tasks from Label Studio")
    all_tasks: List[Dict[str, Any]] = []
    try:
        project_task_count = ls_client.get_task_count(args.project_id)
        total_tasks = project_task_count
        if args.max_tasks > 0:
            total_tasks = min(project_task_count, args.max_tasks)
        print(f"{Colors.CYAN}ℹ Project has {project_task_count} total tasks{Colors.ENDC}")
        
        if args.max_tasks > 0:
            print(f"{Colors.CYAN}ℹ Will fetch up to {args.max_tasks} tasks (--max-tasks){Colors.ENDC}")

        if args.preload_tasks or args.no_batch:
            print(f"{Colors.CYAN}ℹ Preloading task metadata...{Colors.ENDC}")
            all_tasks = ls_client.get_all_tasks(args.project_id, max_tasks=args.max_tasks, show_progress=True)
            total_tasks = len(all_tasks)
            print(f"{Colors.GREEN}✓ Fetched {len(all_tasks)} tasks successfully!{Colors.ENDC}")
        else:
            print(f"{Colors.CYAN}ℹ Streaming task batches; download + labeling starts immediately.{Colors.ENDC}")
            
    except Exception as e:
        print(f"{Colors.RED}✗ Failed to fetch tasks: {e}{Colors.ENDC}")
        sys.exit(1)
    
    if total_tasks == 0:
        print(f"{Colors.YELLOW}⚠ No tasks found in project{Colors.ENDC}")
        sys.exit(0)
    
    # Prepare output directory
    if output_dir.exists():
        if args.force:
            shutil.rmtree(output_dir)
        else:
            print(f"{Colors.RED}✗ Output directory exists. Use --force to overwrite{Colors.ENDC}")
            sys.exit(1)
    
    # Create preview directory if requested
    preview_dir = None
    if args.save_preview:
        preview_dir = Path.home() / "labeled_previews"
        if preview_dir.exists():
            shutil.rmtree(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        print(f"{Colors.GREEN}✓ Preview images will be saved to: {preview_dir}{Colors.ENDC}")
        
        if not PIL_AVAILABLE:
            print(f"{Colors.YELLOW}⚠ PIL/Pillow not installed. Install with: pip install Pillow{Colors.ENDC}")
            preview_dir = None
    
    # Create temp directories for processing
    temp_images_dir = output_dir / "_temp_images"
    temp_labels_dir = output_dir / "_temp_labels"
    try:
        temp_images_dir.mkdir(parents=True, exist_ok=True)
        temp_labels_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        print(f"{Colors.RED}✗ Cannot create output directory: {output_dir}{Colors.ENDC}")
        print(f"  {e}")
        print("  Fix ownership/permissions or choose a different --output-dir.")
        sys.exit(1)
    
    # Process tasks
    print_section("Processing Tasks" + (" (Batch Mode)" if not args.no_batch else " (Sequential)"))
    if args.auto_label:
        concepts_preview = ", ".join(labels[:5])
        if len(labels) > 5:
            concepts_preview += "..."
        print(f"{Colors.CYAN}ℹ Auto-labeling with {len(labels)} concepts: {concepts_preview}{Colors.ENDC}")
    
    if not args.no_batch:
        print(f"{Colors.CYAN}ℹ Batch size: {args.batch_size}, SAM3 batch size: {args.sam3_batch_size}, Workers: {args.workers}{Colors.ENDC}")
    
    stats = Stats()
    stats.total_tasks = total_tasks
    
    print()  # Empty line for progress bar
    
    if args.no_batch:
        # Legacy one-by-one processing (uses pre-fetched tasks)
        tasks_processed = 0
        for task in all_tasks:
            task_id = task.get('id', 'unknown')
            
            success, reason = process_task(
                task,
                ls_client,
                sam3_client,
                temp_images_dir,
                temp_labels_dir,
                label_to_id,
                args.use_existing,
                args.auto_label,
                stats,
                project_labels=labels,
                preview_dir=preview_dir,
            )
            
            tasks_processed += 1
            if success:
                stats.increment_success()
            elif reason in ("no_annotations", "no_valid_boxes", "no_image_url", "download_failed", "invalid_url"):
                stats.increment_skipped(reason)
            elif reason and reason.startswith("sam3_"):
                # SAM3 related errors - these are failures, not skips
                stats.increment_failed(reason)
                print(f"\n  {Colors.RED}✗ Task {task_id}: {reason}{Colors.ENDC}")
            else:
                stats.increment_failed(reason or "unknown")
                if reason:
                    print(f"\n  {Colors.RED}✗ Task {task_id}: {reason}{Colors.ENDC}")
            
            stats.print_progress(task_id)
    else:
        # Batch processing mode - stream task batches by default so fetch/download/labeling overlap by page.
        batch_num = 0
        batch_size = args.batch_size

        if args.preload_tasks:
            batch_iter = (
                all_tasks[i:i + batch_size]
                for i in range(0, len(all_tasks), batch_size)
            )
        else:
            batch_iter = ls_client.iter_task_batches(args.project_id, batch_size=batch_size)

        streamed_tasks = 0
        for batch in batch_iter:
            if args.max_tasks > 0:
                remaining = args.max_tasks - streamed_tasks
                if remaining <= 0:
                    break
                batch = batch[:remaining]
            if not batch:
                break

            streamed_tasks += len(batch)
            batch_num += 1
            batch_info = f"Batch {batch_num} ({len(batch)} tasks)"
            
            process_batch(
                batch,
                ls_client,
                sam3_client,
                temp_images_dir,
                temp_labels_dir,
                label_to_id,
                args.use_existing,
                args.auto_label,
                stats,
                project_labels=labels,
                preview_dir=preview_dir,
                num_workers=args.workers,
                sam3_batch_size=args.sam3_batch_size,
            )
            
            stats.print_progress(batch_info=batch_info)
    
    print()  # New line after progress bar
    
    # Print processing summary
    stats.print_summary()

    real_class_counts = dict(stats.class_counts)
    real_class_image_counts = dict(stats.class_image_counts)
    weak_classes: List[Dict[str, Any]] = []
    prompts_path: Optional[Path] = None
    synthetic_report: Optional[Dict[str, Any]] = None

    if args.balance_classes:
        weak_classes = identify_weak_classes(
            labels,
            real_class_image_counts,
            args.min_samples_per_class,
            args.synthetic_max_ratio,
            args.cosmos_frames_per_video,
        )
        print_section("Class Balance")
        if weak_classes:
            print(f"{Colors.YELLOW}⚠ {len(weak_classes)} class(es) below {args.min_samples_per_class} images:{Colors.ENDC}")
            for weak in weak_classes:
                print(
                    f"    {weak['class_name']}: {weak['current_count']} images, "
                    f"need {weak['deficit']}, planned synthetic frames <= {weak['target_synthetic_frames']}"
                )
        else:
            print(f"{Colors.GREEN}✓ All classes meet the minimum sample target{Colors.ENDC}")
    
    # Split dataset
    print_section("Splitting Dataset")
    counts = split_dataset(
        temp_images_dir,
        temp_labels_dir,
        output_dir,
        args.train_split,
        args.val_split,
        args.test_split,
    )
    
    # Clean up temp directories
    shutil.rmtree(temp_images_dir, ignore_errors=True)
    shutil.rmtree(temp_labels_dir, ignore_errors=True)

    if args.cosmos_augment and weak_classes:
        print_section("Cosmos Synthetic Augmentation")
        project_title = project.get("title", f"project_{args.project_id}")
        prompts_path = project_path(args.cosmos_prompts_out, args.project_id)
        jobs_dir = project_path(args.cosmos_jobs_dir, args.project_id)
        cosmos_output_dir = project_path(args.cosmos_output_dir, args.project_id)
        frames_dir = project_path(args.cosmos_frames_dir, args.project_id)

        try:
            if args.lmstudio_model:
                print(f"{Colors.CYAN}ℹ Generating project-specific Cosmos prompts with LM Studio...{Colors.ENDC}")
                lm_client = LMStudioPromptClient(
                    args.lmstudio_url,
                    args.lmstudio_api_key,
                    args.lmstudio_model,
                    args.lmstudio_timeout,
                )
                cosmos_prompts = lm_client.generate_prompts(
                    project_title,
                    labels,
                    weak_classes,
                    args.cosmos_frames_per_video,
                )
            else:
                print(f"{Colors.YELLOW}⚠ LM Studio model not set; using deterministic fallback prompts{Colors.ENDC}")
                cosmos_prompts = fallback_cosmos_prompts(project_title, weak_classes)
        except Exception as e:
            print(f"{Colors.YELLOW}⚠ LM Studio prompt generation failed: {e}{Colors.ENDC}")
            print(f"{Colors.YELLOW}  Falling back to deterministic prompts so the loop can continue.{Colors.ENDC}")
            cosmos_prompts = fallback_cosmos_prompts(project_title, weak_classes)

        write_json_file(prompts_path, cosmos_prompts)
        print(f"{Colors.GREEN}✓ Saved Cosmos prompts: {prompts_path}{Colors.ENDC}")

        try:
            run_cosmos_prompt_batch(
                prompts_path,
                jobs_dir,
                cosmos_output_dir,
                frames_dir,
                args.cosmos_model,
                args.cosmos_frames_per_video,
                args.cosmos_build,
            )
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}✗ Cosmos generation failed with exit code {e.returncode}{Colors.ENDC}")
            print("  Fix the Cosmos Docker run, then rerun with --cosmos-augment.")
            write_quality_report(
                output_dir,
                project,
                labels,
                real_class_counts,
                dict(stats.skipped_reasons),
                dict(stats.failed_reasons),
                weak_classes,
                prompts_path,
                None,
            )
            sys.exit(1)

        print(f"{Colors.CYAN}ℹ Labeling Cosmos frames with SAM3...{Colors.ENDC}")
        synthetic_report = label_synthetic_frames(
            frames_dir,
            jobs_dir,
            output_dir,
            label_to_id,
            sam3_client,
            preview_dir,
        )
        counts["train"] += int(synthetic_report["accepted_images"])
        print(
            f"{Colors.GREEN}✓ Accepted {synthetic_report['accepted_images']} synthetic train images "
            f"with {synthetic_report['accepted_boxes']} boxes{Colors.ENDC}"
        )
    elif args.cosmos_augment:
        print_section("Cosmos Synthetic Augmentation")
        print(f"{Colors.GREEN}✓ Skipped: no weak classes found{Colors.ENDC}")

    if args.balance_classes or args.cosmos_augment:
        write_quality_report(
            output_dir,
            project,
            labels,
            real_class_counts,
            dict(stats.skipped_reasons),
            dict(stats.failed_reasons),
            weak_classes,
            prompts_path,
            synthetic_report,
        )
        print(f"{Colors.GREEN}✓ Wrote quality report: {output_dir / 'quality_report.json'}{Colors.ENDC}")
    
    # Create dataset.yaml
    create_dataset_yaml(output_dir, labels, project.get("title", "yolo_dataset"))
    
    # Print final summary
    print_section("Dataset Ready!")
    print(f"""
  Output Directory: {output_dir.absolute()}
  
  Dataset Split:
    • Train: {counts['train']} images
    • Val:   {counts['val']} images  
    • Test:  {counts['test']} images
    • Total: {sum(counts.values())} images
  
  Classes ({len(labels)}):""")
    
    for i, label in enumerate(labels):
        print(f"    {i}: {label}")
    
    print(f"""
  {Colors.GREEN}To train YOLO:{Colors.ENDC}
  yolo detect train data={output_dir.absolute()}/dataset.yaml model=yolov8n.pt epochs=100
""")


if __name__ == "__main__":
    main()
