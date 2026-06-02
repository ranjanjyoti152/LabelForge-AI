from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import ContextManager, Dict, Iterable, List, Sequence
import importlib.util
import os
import shutil
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from app.core.config import Settings


def _get_bpe_path() -> str | None:
    """Get the path to the BPE vocabulary file from various packages.
    
    The BPE vocabulary file is shared across multiple packages (sam3, clip, open_clip).
    We try to find it from any available source.
    """
    search_packages = ["sam3", "clip", "open_clip"]
    
    for package_name in search_packages:
        try:
            spec = importlib.util.find_spec(package_name)
            if spec is None:
                continue
            
            # Handle namespace packages (origin is None)
            if spec.origin:
                package_dir = Path(spec.origin).resolve().parent
            elif hasattr(spec, 'submodule_search_locations') and spec.submodule_search_locations:
                # Use the first search location for namespace packages
                package_dir = Path(list(spec.submodule_search_locations)[0])
            else:
                continue
            
            # Check for the BPE file in common locations
            bpe_paths = [
                package_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz",
                package_dir / "bpe_simple_vocab_16e6.txt.gz",
            ]
            
            for bpe_path in bpe_paths:
                if bpe_path.exists():
                    return str(bpe_path)
        except Exception:
            continue
    
    return None


def _ensure_directories() -> None:
    """Create required directories if they don't exist."""
    dirs = [
        Path("./weights"),
        Path("./logs"),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"[INIT] Directory ensured: {d}")


def _is_missing_sam3_sam(exc: ModuleNotFoundError) -> bool:
    missing = getattr(exc, "name", "") or ""
    return missing == "sam3.sam" or missing.startswith("sam3.sam.")


def _extend_sam3_namespace() -> None:
    spec = importlib.util.find_spec("sam3")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "sam3 package is not installed. Install facebookresearch/sam3 first."
        )

    package_dir = Path(spec.origin).resolve().parent
    target_dir = package_dir / "sam"
    if target_dir.exists() and any(target_dir.iterdir()):
        return

    vendor_dir = Path(__file__).resolve().parents[2] / "third_party" / "sam3_patch" / "sam3" / "sam"
    if not vendor_dir.exists():
        raise RuntimeError(
            "sam3.sam is missing from the published wheel and no vendored copy was found."
        )

    shutil.copytree(vendor_dir, target_dir, dirs_exist_ok=True)


def _load_sam3_bindings():
    try:
        from sam3.model_builder import build_sam3_image_model  # type: ignore
        from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore
        return build_sam3_image_model, Sam3Processor
    except ModuleNotFoundError as exc:
        if _is_missing_sam3_sam(exc):
            _extend_sam3_namespace()
            from sam3.model_builder import build_sam3_image_model  # type: ignore
            from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore
            return build_sam3_image_model, Sam3Processor
        raise


try:
    build_sam3_image_model, Sam3Processor = _load_sam3_bindings()
except ImportError as exc:  # pragma: no cover - optional dependency at runtime
    Sam3Processor = None  # type: ignore
    build_sam3_image_model = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


COCO_CONCEPTS: List[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

SAFETY_CONCEPTS: List[str] = [
    "fire",
    "flames",
    "smoke",
    "spark",
    "explosion",
    "steam leak",
    "gas leak",
    "chemical spill",
    "safety helmet",
    "hard hat",
    "helmet",
    "safety goggles",
    "protective gloves",
    "safety gloves",
    "face shield",
    "respirator mask",
    "safety mask",
    "safety vest",
    "reflective vest",
    "life vest",
    "safety harness",
    "fall arrest system",
    "traffic cone",
    "warning sign",
    "danger sign",
    "barrier",
    "safety barricade",
    "fire extinguisher",
    "sprinkler",
    "alarm beacon",
    "first aid kit",
    "emergency exit",
    "forklift",
    "crane",
    "bulldozer",
    "excavator",
    "dump truck",
    "loader",
    "pipe leak",
    "welding arc",
    "hot surface",
    "electrical panel",
    "battery pack",
    "overhead line",
]

# Additional concepts for Indian traffic / Label Studio projects
CUSTOM_CONCEPTS: List[str] = [
    "auto rickshaw",
    "rickshaw",
    "ambulance",
    "license plate",
    "number plate",
    "License_Plate",
]

DEFAULT_CONCEPTS: List[str] = COCO_CONCEPTS + SAFETY_CONCEPTS + CUSTOM_CONCEPTS


@dataclass(slots=True)
class Detection:
    frame_index: int
    label: str
    bbox: Dict[str, int]
    area: int
    score: float


class Sam3Detector:
    """Text-prompted SAM3 detector that returns per-concept bounding boxes."""

    def __init__(self, settings: Settings) -> None:
        # Ensure required directories exist
        _ensure_directories()
        
        self.settings = settings
        preferred = settings.device.lower()
        if preferred == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(preferred)

        self._processor: Sam3Processor | None = None
        self._default_concepts = self._build_default_concepts()
        self._build_processor()

        # Enable inference optimizations
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _build_processor(self) -> None:
        if Sam3Processor is None or build_sam3_image_model is None:
            raise RuntimeError(
                "sam3 package missing. Install facebookresearch/sam3 before running detections."
            ) from _IMPORT_ERROR

        if self.settings.hf_token and "HF_TOKEN" not in os.environ:
            os.environ["HF_TOKEN"] = self.settings.hf_token
            os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", self.settings.hf_token)

        # Get BPE path explicitly to avoid pkg_resources issue
        bpe_path = _get_bpe_path()
        if bpe_path:
            print(f"[INIT] Found BPE vocabulary at: {bpe_path}")
        else:
            print("[INIT] Warning: Could not locate BPE vocabulary file")

        checkpoint = self.settings.checkpoint_path
        checkpoint_path = Path(checkpoint) if checkpoint else None
        
        # Check if checkpoint exists, if not download from HuggingFace
        load_from_hf = checkpoint_path is None or not checkpoint_path.exists()
        
        if load_from_hf:
            print("[INIT] Model checkpoint not found locally. Downloading from HuggingFace...")
            print("[INIT] This may take a while on first run (model is ~2GB)...")
            if checkpoint_path:
                print(f"[INIT] Will save to: {checkpoint_path}")
            # Enable tqdm progress bars for huggingface_hub downloads
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            print(f"[INIT] Loading model from local checkpoint: {checkpoint_path}")
        
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=checkpoint if not load_from_hf else None,
            load_from_HF=load_from_hf,
            device=str(self.device),
            eval_mode=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
        
        print(f"[INIT] SAM3 model loaded successfully on {self.device}")
        
        self._processor = Sam3Processor(
            model=model,
            device=str(self.device),
            confidence_threshold=self.settings.score_threshold,
        )

    def _build_default_concepts(self) -> List[str]:
        concepts: List[str] = []
        if self.settings.concepts_path:
            concepts.extend(self._read_concepts_from_file(self.settings.concepts_path))
        elif self.settings.use_default_concepts:
            concepts.extend(DEFAULT_CONCEPTS)

        if self.settings.extra_concepts:
            concepts.extend(self._split_extra_prompts(self.settings.extra_concepts))

        deduped = self._dedupe(concepts)
        # Allow empty concepts - they will be provided at runtime from Label Studio
        return deduped

    def _read_concepts_from_file(self, path: str) -> List[str]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Concepts file not found: {file_path}")

        concepts: List[str] = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                cleaned = line.strip()
                if cleaned:
                    concepts.append(cleaned)
        return concepts

    def _split_extra_prompts(self, raw_prompts: str) -> List[str]:
        return [prompt.strip() for prompt in raw_prompts.split(",") if prompt.strip()]

    def _dedupe(self, values: Iterable[str]) -> List[str]:
        seen: Dict[str, None] = {}
        for value in values:
            key = value.strip()
            if not key or key in seen:
                continue
            seen[key] = None
        return list(seen.keys())

    def _prepare_concepts(self, overrides: Sequence[str] | None) -> List[str]:
        prompts = [prompt.strip() for prompt in (overrides or self._default_concepts) if prompt.strip()]
        if not prompts:
            raise ValueError("At least one concept prompt must be configured before running detection.")
        return prompts

    def _inference_autocast(self) -> ContextManager[None]:
        """SAM3's fused ViT path emits bf16 activations; autocast keeps later layers compatible."""
        if self.device.type in {"cuda", "cpu"}:
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return nullcontext()

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int,
        concepts: Sequence[str] | None = None,
    ) -> List[Detection]:
        if self._processor is None:
            raise RuntimeError("SAM3 processor has not been initialized.")

        prompts = self._prepare_concepts(concepts)
        
        # Convert frame once
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Set image once - this is the expensive operation
        with torch.inference_mode(), self._inference_autocast():
            state = self._processor.set_image(pil_frame)

            detections: List[Detection] = []
            score_threshold = self.settings.score_threshold
            max_detections = self.settings.max_detections
            
            for prompt in prompts:
                state = self._processor.set_text_prompt(prompt=prompt, state=state)
                boxes = state.get("boxes")
                scores = state.get("scores")
                
                if boxes is None or scores is None:
                    self._processor.reset_all_prompts(state)
                    continue

                # Fast path: skip mask computation if no valid detections
                if scores.numel() == 0 or scores.max().item() < score_threshold:
                    self._processor.reset_all_prompts(state)
                    continue
                
                masks = state.get("masks")
                if masks is None:
                    self._processor.reset_all_prompts(state)
                    continue

                # Don't limit during detection - collect all, filter with NMS later
                detections.extend(
                    self._serialize_detections_fast(
                        prompt=prompt,
                        frame_index=frame_index,
                        boxes=boxes,
                        scores=scores,
                        masks=masks,
                        limit=50,  # Per-prompt limit to avoid memory issues
                        score_threshold=score_threshold,
                    )
                )
                self._processor.reset_all_prompts(state)

        # Filter tiny boxes (likely false positives)
        min_box_area = getattr(self.settings, 'min_box_area', 500)
        if min_box_area > 0:
            detections = [d for d in detections if d.bbox["width"] * d.bbox["height"] >= min_box_area]

        # Apply NMS to remove duplicate detections
        nms_threshold = getattr(self.settings, 'nms_threshold', 0.3)
        cross_class_nms = getattr(self.settings, 'cross_class_nms', True)
        
        # First apply per-class NMS
        detections = self._apply_nms_per_class(detections, iou_threshold=nms_threshold)
        
        # Then apply cross-class NMS to remove overlapping boxes of different labels on same object
        if cross_class_nms:
            detections = self._apply_cross_class_nms(detections, iou_threshold=nms_threshold)
        
        # Apply max_detections limit after NMS (keep highest scoring)
        if len(detections) > max_detections:
            detections = sorted(detections, key=lambda d: d.score, reverse=True)[:max_detections]
        
        return detections

    def _apply_cross_class_nms(self, detections: List[Detection], iou_threshold: float = 0.3) -> List[Detection]:
        """Apply NMS across ALL classes - removes overlapping boxes even with different labels.
        
        This handles cases where the same object is detected with different prompts
        (e.g., 'car' and 'truck' on the same vehicle). Keep the higher scoring one.
        """
        if len(detections) <= 1:
            return detections
        
        # Sort by score descending - highest score wins
        detections = sorted(detections, key=lambda d: d.score, reverse=True)
        
        keep: List[Detection] = []
        for det in detections:
            # Check if this detection overlaps too much with any already-kept detection
            dominated = False
            for kept in keep:
                iou = self._compute_iou(det.bbox, kept.bbox)
                if iou > iou_threshold:
                    # This box overlaps with a higher-scoring box, skip it
                    dominated = True
                    break
            if not dominated:
                keep.append(det)
        
        return keep

    def _apply_nms_per_class(self, detections: List[Detection], iou_threshold: float = 0.7) -> List[Detection]:
        """Apply NMS per class - only suppress overlapping boxes of the SAME label."""
        if len(detections) <= 1:
            return detections
        
        # Group by label
        by_label: Dict[str, List[Detection]] = {}
        for det in detections:
            label = det.label.lower()
            if label not in by_label:
                by_label[label] = []
            by_label[label].append(det)
        
        # Apply NMS within each label group
        keep: List[Detection] = []
        for label, dets in by_label.items():
            # Sort by score descending
            dets = sorted(dets, key=lambda d: d.score, reverse=True)
            label_keep: List[Detection] = []
            for det in dets:
                dominated = False
                for kept in label_keep:
                    iou = self._compute_iou(det.bbox, kept.bbox)
                    if iou > iou_threshold:
                        dominated = True
                        break
                if not dominated:
                    label_keep.append(det)
            keep.extend(label_keep)
        
        # Sort final results by score
        keep.sort(key=lambda d: d.score, reverse=True)
        return keep

    def _apply_nms(self, detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
        """Apply Non-Maximum Suppression to remove overlapping detections."""
        if len(detections) <= 1:
            return detections
        
        # Sort by score descending
        detections = sorted(detections, key=lambda d: d.score, reverse=True)
        
        keep: List[Detection] = []
        for det in detections:
            # Check if this detection overlaps too much with any kept detection
            dominated = False
            for kept in keep:
                iou = self._compute_iou(det.bbox, kept.bbox)
                if iou > iou_threshold:
                    dominated = True
                    break
            if not dominated:
                keep.append(det)
        
        return keep
    
    def _compute_iou(self, box1: Dict[str, int], box2: Dict[str, int]) -> float:
        """Compute Intersection over Union between two boxes."""
        x1_1, y1_1 = box1["x"], box1["y"]
        x2_1, y2_1 = x1_1 + box1["width"], y1_1 + box1["height"]
        
        x1_2, y1_2 = box2["x"], box2["y"]
        x2_2, y2_2 = x1_2 + box2["width"], y1_2 + box2["height"]
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        # Union
        area1 = box1["width"] * box1["height"]
        area2 = box2["width"] * box2["height"]
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area

    def _serialize_detections_fast(
        self,
        prompt: str,
        frame_index: int,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        masks: torch.Tensor,
        limit: int,
        score_threshold: float,
    ) -> List[Detection]:
        """Optimized serialization with batched operations."""
        # Filter by score first on GPU
        valid_mask = scores >= score_threshold
        if not valid_mask.any():
            return []
        
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        count = min(limit, len(valid_indices))
        if count == 0:
            return []
        
        # Batch transfer to CPU
        valid_indices = valid_indices[:count]
        boxes_cpu = boxes[valid_indices].detach().cpu()
        scores_cpu = scores[valid_indices].detach().cpu()
        masks_cpu = masks[valid_indices].detach().cpu().numpy()
        
        serialized: List[Detection] = []
        for idx in range(count):
            score = float(scores_cpu[idx].item())
            box = boxes_cpu[idx].tolist()
            
            x0 = max(0, int(round(box[0])))
            y0 = max(0, int(round(box[1])))
            x1 = max(x0 + 1, int(round(box[2])))
            y1 = max(y0 + 1, int(round(box[3])))

            width = x1 - x0
            height = y1 - y0

            # Fast area calculation
            area = int(masks_cpu[idx].sum())
            
            serialized.append(
                Detection(
                    frame_index=frame_index,
                    label=prompt,
                    bbox={"x": x0, "y": y0, "width": width, "height": height},
                    area=area if area > 0 else width * height,
                    score=score,
                )
            )
        return serialized
