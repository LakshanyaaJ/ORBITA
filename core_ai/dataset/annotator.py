"""
ORBITA Assisted Annotation & Verification Pipeline
==================================================
Provides model-assisted pre-labeling with strict verification status gates.
Pre-labels frames using existing detection models and maps bounding boxes
to the standard ORBITA ontology.

Status Workflow:
  UNANNOTATED -> PRE_ANNOTATED (Assisted) -> VERIFIED (Human confirmed)

Safety Rules:
  1. The system NEVER silently trains on unverified pseudo-labels.
  2. Each label file (.txt) has a sidecar (.meta.json) tracking provenance,
     detection confidence, and human verification status.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core_ai.dataset.dataset_manager import ORBITA_CLASSES

logger = logging.getLogger(__name__)

CLASS_TO_ID = {name: idx for idx, name in enumerate(ORBITA_CLASSES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(ORBITA_CLASSES)}


class AnnotationStatus(str, Enum):
    UNANNOTATED = "unannotated"
    PRE_ANNOTATED = "pre_annotated"
    VERIFIED = "verified"
    REJECTED = "rejected"


@dataclass
class BoundingBoxLabel:
    """YOLO-format normalized bounding box."""
    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float = 1.0


@dataclass
class AnnotationRecord:
    """Annotation metadata for a single frame."""
    image_name: str
    image_path: str
    label_path: str
    meta_path: str
    status: AnnotationStatus
    boxes: list[BoundingBoxLabel] = field(default_factory=list)
    annotated_by: str = "assistant"
    verified_by: Optional[str] = None
    verification_notes: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class AssistedAnnotator:
    """
    Generates YOLO pre-annotations using existing ORBITA perception models.
    """

    def __init__(self, detector=None):
        self.detector = detector
        if self.detector is None:
            # Lazy load ObjectDetector with default configuration if available
            try:
                from core_ai.app.config import load_config
                from core_ai.perception.object_detector import ObjectDetector
                cfg = load_config()
                self.detector = ObjectDetector(cfg.detection)
            except Exception as exc:
                logger.warning("ObjectDetector unavailable for assisted annotation: %s", exc)

    def pre_annotate_image(
        self,
        image_path: str | Path,
        label_output_path: str | Path,
        min_confidence: float = 0.35,
    ) -> AnnotationRecord:
        """
        Run detector on image and generate pre-annotated YOLO .txt and .meta.json files.
        """
        img_path = Path(image_path)
        lbl_path = Path(label_output_path)
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path = lbl_path.with_suffix(".meta.json")

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")

        h, w = img.shape[:2]
        boxes: list[BoundingBoxLabel] = []

        if self.detector is not None:
            try:
                detections = self.detector.detect(img)
                for det in detections:
                    if det.confidence < min_confidence:
                        continue

                    class_name = det.class_name.upper()
                    if class_name not in CLASS_TO_ID:
                        continue

                    class_id = CLASS_TO_ID[class_name]
                    bx, by, bw, bh = det.bbox

                    # Convert pixel (x, y, w, h) to normalized YOLO (x_center, y_center, w, h)
                    x_center = (bx + bw / 2.0) / w
                    y_center = (by + bh / 2.0) / h
                    norm_w = bw / w
                    norm_h = bh / h

                    # Clamp to [0, 1]
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    norm_w = max(0.0, min(1.0, norm_w))
                    norm_h = max(0.0, min(1.0, norm_h))

                    boxes.append(BoundingBoxLabel(
                        class_id=class_id,
                        class_name=class_name,
                        x_center=round(x_center, 6),
                        y_center=round(y_center, 6),
                        width=round(norm_w, 6),
                        height=round(norm_h, 6),
                        confidence=round(det.confidence, 3),
                    ))
            except Exception as exc:
                logger.warning("Error running detector on %s: %s", img_path.name, exc)

        # Write YOLO format .txt
        txt_lines = [
            f"{b.class_id} {b.x_center:.6f} {b.y_center:.6f} {b.width:.6f} {b.height:.6f}"
            for b in boxes
        ]
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines) + ("\n" if txt_lines else ""))

        status = AnnotationStatus.PRE_ANNOTATED if boxes else AnnotationStatus.UNANNOTATED

        record = AnnotationRecord(
            image_name=img_path.name,
            image_path=str(img_path.resolve()),
            label_path=str(lbl_path.resolve()),
            meta_path=str(meta_path.resolve()),
            status=status,
            boxes=boxes,
            annotated_by="orbita_yolo_assistant",
            verified_by=None,
        )

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)

        return record

    def verify_annotation(
        self,
        meta_path: str | Path,
        verified_by: str = "human_reviewer",
        notes: Optional[str] = None,
        updated_boxes: Optional[list[dict]] = None,
    ) -> AnnotationRecord:
        """
        Marks an annotation as human-verified, optionally updating the bounding box coordinates.
        """
        meta_p = Path(meta_path)
        with open(meta_p, "r", encoding="utf-8") as f:
            data = json.load(f)

        lbl_p = Path(data["label_path"])

        if updated_boxes is not None:
            # Overwrite .txt with updated boxes
            txt_lines = []
            box_objs = []
            for b in updated_boxes:
                cid = int(b["class_id"])
                xc = float(b["x_center"])
                yc = float(b["y_center"])
                w = float(b["width"])
                h = float(b["height"])
                cname = ID_TO_CLASS.get(cid, "UNKNOWN")
                txt_lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                box_objs.append(BoundingBoxLabel(
                    class_id=cid,
                    class_name=cname,
                    x_center=xc,
                    y_center=yc,
                    width=w,
                    height=h,
                    confidence=1.0,
                ))
            with open(lbl_p, "w", encoding="utf-8") as f:
                f.write("\n".join(txt_lines) + ("\n" if txt_lines else ""))
            data["boxes"] = [asdict(b) for b in box_objs]

        data["status"] = AnnotationStatus.VERIFIED.value
        data["verified_by"] = verified_by
        data["verification_notes"] = notes

        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        boxes = [BoundingBoxLabel(**b) for b in data.get("boxes", [])]
        return AnnotationRecord(
            image_name=data["image_name"],
            image_path=data["image_path"],
            label_path=data["label_path"],
            meta_path=data["meta_path"],
            status=AnnotationStatus.VERIFIED,
            boxes=boxes,
            annotated_by=data.get("annotated_by", "assistant"),
            verified_by=verified_by,
            verification_notes=notes,
        )
