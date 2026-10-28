from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from src.models.model_loader import Detection

@dataclass
class SafetyAssessment:
    """Represents safety evaluation for an individual worker."""
    worker: Detection
    has_helmet: bool
    has_vest: bool
    helmet_detection: Optional[Detection] = None
    vest_detection: Optional[Detection] = None
    is_violation: bool = False
    violation_reasons: List[str] = None
    uncertain: bool = False
    confidence: float = 1.0

    def __post_init__(self):
        if self.violation_reasons is None:
            self.violation_reasons = []

def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0:
        return 0.0
    return interArea / unionArea

def calculate_containment(inner: List[float], outer: List[float]) -> float:
    """Computes the fraction of inner box area contained inside outer box."""
    xA = max(inner[0], outer[0])
    yA = max(inner[1], outer[1])
    xB = min(inner[2], outer[2])
    yB = min(inner[3], outer[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    innerArea = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])

    if innerArea <= 0:
        return 0.0
    return interArea / innerArea

class PPERuleEngine:
    """Decoupled safety rule engine evaluating PPE compliance per Section 4.5."""

    def __init__(
        self,
        require_helmet: bool = True,
        require_vest: bool = True,
        helmet_conf: float = 0.35,
        vest_conf: float = 0.35,
        uncertain_thresh: float = 0.25
    ):
        self.require_helmet = require_helmet
        self.require_vest = require_vest
        self.helmet_conf = helmet_conf
        self.vest_conf = vest_conf
        self.uncertain_thresh = uncertain_thresh

    def assess_worker(self, worker: Detection, detections: List[Detection]) -> SafetyAssessment:
        """Associates helmet, vest, and head detections to a specific worker using spatial heuristics."""
        wx1, wy1, wx2, wy2 = worker.box
        w_height = worker.height
        w_width = worker.width

        # Worker head sub-region: upper 35% of worker bounding box
        head_region = [wx1 - 0.1 * w_width, wy1 - 0.1 * w_height, wx2 + 0.1 * w_width, wy1 + 0.40 * w_height]
        # Worker torso sub-region: middle 20% to 75% of worker bounding box
        torso_region = [wx1 - 0.15 * w_width, wy1 + 0.15 * w_height, wx2 + 0.15 * w_width, wy1 + 0.80 * w_height]

        associated_helmet: Optional[Detection] = None
        best_helmet_score = 0.0

        associated_vest: Optional[Detection] = None
        best_vest_score = 0.0

        has_explicit_no_helmet = False
        has_explicit_no_vest = False

        for det in detections:
            if det == worker:
                continue

            # Check explicit negative classes (if trained on dataset with 'no-helmet' / 'no-vest')
            if det.class_name == "no-helmet" and calculate_containment(det.box, head_region) > 0.3:
                has_explicit_no_helmet = True
            elif det.class_name == "no-vest" and calculate_containment(det.box, torso_region) > 0.3:
                has_explicit_no_vest = True

            # Check positive helmet
            elif det.class_name == "helmet":
                containment = calculate_containment(det.box, head_region)
                iou = calculate_iou(det.box, head_region)
                score = max(containment, iou)
                if score > 0.15 and det.confidence > best_helmet_score:
                    best_helmet_score = det.confidence
                    associated_helmet = det

            # Check positive vest
            elif det.class_name == "vest":
                containment = calculate_containment(det.box, torso_region)
                iou = calculate_iou(det.box, torso_region)
                score = max(containment, iou)
                if score > 0.15 and det.confidence > best_vest_score:
                    best_vest_score = det.confidence
                    associated_vest = det

        has_helmet = (associated_helmet is not None) and (not has_explicit_no_helmet)
        has_vest = (associated_vest is not None) and (not has_explicit_no_vest)

        violations = []
        is_violation = False
        uncertain = False

        if self.require_helmet:
            if not has_helmet:
                violations.append("Missing Helmet")
                is_violation = True
            elif associated_helmet and associated_helmet.confidence < self.uncertain_thresh:
                uncertain = True

        if self.require_vest:
            if not has_vest:
                violations.append("Missing Safety Vest")
                is_violation = True
            elif associated_vest and associated_vest.confidence < self.uncertain_thresh:
                uncertain = True

        # Calculate overall assessment confidence
        conf_values = [worker.confidence]
        if associated_helmet:
            conf_values.append(associated_helmet.confidence)
        if associated_vest:
            conf_values.append(associated_vest.confidence)
            
        avg_confidence = float(np.mean(conf_values))

        return SafetyAssessment(
            worker=worker,
            has_helmet=has_helmet,
            has_vest=has_vest,
            helmet_detection=associated_helmet,
            vest_detection=associated_vest,
            is_violation=is_violation,
            violation_reasons=violations,
            uncertain=uncertain,
            confidence=avg_confidence
        )

    def evaluate_scene(self, detections: List[Detection]) -> List[SafetyAssessment]:
        """Evaluates all detected workers in an image against configured safety rules."""
        workers = [d for d in detections if d.class_name in ("person", "worker")]
        assessments: List[SafetyAssessment] = []

        for worker in workers:
            assessment = self.assess_worker(worker, detections)
            assessments.append(assessment)

        return assessments

    def annotate_safety(
        self,
        image: np.ndarray,
        assessments: List[SafetyAssessment],
        line_thickness: int = 2
    ) -> np.ndarray:
        """Draws human-auditable safety assessment boxes and violation indicators on the image."""
        import cv2
        annotated = image.copy()

        for a in assessments:
            x1, y1, x2, y2 = map(int, a.worker.box)

            if a.uncertain:
                color = (0, 255, 255) # Yellow
                status_text = "NEEDS REVIEW"
            elif a.is_violation:
                color = (0, 0, 255)   # Red
                status_text = f"VIOLATION: {', '.join(a.violation_reasons)}"
            else:
                color = (0, 220, 0)   # Green
                status_text = "COMPLIANT"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

            label = f"{status_text} ({a.confidence:.2f})"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - text_h - 6)), (x1 + text_w + 6, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, max(0, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return annotated
