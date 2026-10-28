from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from ultralytics import YOLO

from src.models.model_loader import Detection
from src.utils.logging_config import setup_logger

logger = setup_logger("detector")

class SafeSightDetector:
    """Wrapper around Ultralytics YOLO detector with class normalization for PPE & Workers."""
    
    # Standardized normalized class targets
    CLASS_WORKER = "person"
    CLASS_HELMET = "helmet"
    CLASS_VEST = "vest"
    CLASS_HEAD = "head"
    CLASS_NO_HELMET = "no-helmet"
    CLASS_NO_VEST = "no-vest"

    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu", conf: float = 0.35, iou: float = 0.45):
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.iou = iou
        self._load_model()

    def _load_model(self) -> None:
        try:
            logger.info(f"Loading YOLO detector from: {self.model_path} on {self.device}")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            # Map raw class IDs to normalized names
            self.class_names = self.model.names if hasattr(self.model, "names") else {}
            logger.info(f"Model loaded successfully with {len(self.class_names)} classes: {self.class_names}")
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise

    def normalize_class_name(self, raw_name: str) -> str:
        """Normalizes dataset-specific class names to unified standard names."""
        name = raw_name.lower().strip()
        if name in ["person", "worker", "human", "people"]:
            return self.CLASS_WORKER
        elif name in ["helmet", "hard-hat", "hardhat", "safety-helmet", "hat"]:
            return self.CLASS_HELMET
        elif name in ["vest", "safety-vest", "reflective-vest", "high-vis-vest"]:
            return self.CLASS_VEST
        elif name in ["head"]:
            return self.CLASS_HEAD
        elif name in ["no-helmet", "no_helmet", "without-helmet"]:
            return self.CLASS_NO_HELMET
        elif name in ["no-vest", "no_vest", "without-vest"]:
            return self.CLASS_NO_VEST
        return name

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Runs object detection on an image (BGR format as returned by cv2)."""
        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )
        
        detections: List[Detection] = []
        if not results:
            return detections

        first_res = results[0]
        if first_res.boxes is None:
            return detections

        boxes = first_res.boxes.xyxy.cpu().numpy()
        confs = first_res.boxes.conf.cpu().numpy()
        classes = first_res.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, classes):
            raw_name = self.class_names.get(cls_id, str(cls_id))
            normalized_name = self.normalize_class_name(raw_name)
            detections.append(
                Detection(
                    class_id=int(cls_id),
                    class_name=normalized_name,
                    confidence=float(conf),
                    box=[float(b) for b in box]
                )
            )

        return detections

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        line_thickness: int = 2
    ) -> np.ndarray:
        """Annotates image with bounding boxes and normalized class labels."""
        annotated = image.copy()
        
        color_palette = {
            self.CLASS_WORKER: (255, 178, 50),     # Bright blue/cyan
            self.CLASS_HELMET: (0, 200, 0),        # Vibrant Green
            self.CLASS_VEST: (0, 165, 255),        # Amber/Orange
            self.CLASS_HEAD: (200, 200, 200),      # Light Gray
            self.CLASS_NO_HELMET: (0, 0, 255),     # Crimson Red
            self.CLASS_NO_VEST: (0, 0, 220),       # Crimson Red
        }
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det.box)
            color = color_palette.get(det.class_name, (200, 100, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)
            
            # Label banner
            label = f"{det.class_name} {det.confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - text_h - 4)), (x1 + text_w, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1, max(0, y1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            
        return annotated
