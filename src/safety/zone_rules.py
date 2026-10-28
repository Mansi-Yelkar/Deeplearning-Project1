from typing import List, Tuple, Optional
import cv2
import numpy as np
from src.models.model_loader import Detection

class RestrictedZoneRuleEngine:
    """Detects worker incursions into user-configured polygonal danger zones."""

    def __init__(self, polygon_points: Optional[List[Tuple[int, int]]] = None):
        """polygon_points: list of (x, y) coordinates defining boundary."""
        self.polygon_points = polygon_points or []

    def is_inside_zone(self, point: Tuple[float, float]) -> bool:
        """Determines if a 2D point (x, y) is inside the polygon using pointPolygonTest."""
        if len(self.polygon_points) < 3:
            return False
        contour = np.array(self.polygon_points, dtype=np.int32)
        dist = cv2.pointPolygonTest(contour, point, False)
        return dist >= 0

    def check_worker_incursion(self, worker: Detection) -> Tuple[bool, str]:
        """Checks if bottom-center of worker box (worker's feet proxy) enters the restricted zone."""
        if len(self.polygon_points) < 3:
            return False, "No active zone defined"

        # Feet location is standard ground-plane proxy in computer vision
        feet_point = ((worker.x1 + worker.x2) / 2.0, worker.y2)
        inside = self.is_inside_zone(feet_point)
        if inside:
            return True, "Restricted Zone Incursion Detected"
        return False, "Zone Clear"

    def draw_zone(self, image: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """Overlays semi-transparent polygon highlighting restricted area."""
        if len(self.polygon_points) < 3:
            return image

        annotated = image.copy()
        overlay = image.copy()
        pts = np.array(self.polygon_points, dtype=np.int32).reshape((-1, 1, 2))
        
        cv2.fillPoly(overlay, [pts], (0, 0, 220)) # Red danger zone
        cv2.polylines(annotated, [pts], True, (0, 0, 255), 2)
        
        cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
        return annotated
