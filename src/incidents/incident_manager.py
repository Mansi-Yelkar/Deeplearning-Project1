from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from src.incidents.database import IncidentDatabase
from src.utils.logging_config import setup_logger

logger = setup_logger("incident_manager")

class IncidentManager:
    """Coordinates incident logging, evidence frame persistence, and retrieval."""
    
    def __init__(self, db_path: str = "incidents.db", evidence_dir: str = "outputs/evidence"):
        self.db = IncidentDatabase(db_path)
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def save_evidence_image(self, image: np.ndarray, prefix: str = "violation") -> str:
        """Saves an annotated evidence frame to disk and returns its relative file path."""
        import uuid
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = self.evidence_dir / filename
        cv2.imwrite(str(filepath), image)
        return str(filepath)

    def log_incident(
        self,
        source_filename: str,
        source_type: str,
        event_category: str,
        confidence: float,
        rule_status: str,
        evidence_image: Optional[np.ndarray] = None,
        timestamp_seconds: Optional[float] = None,
        reviewer_notes: str = ""
    ) -> int:
        """Logs an incident, saving evidence frame if provided, and persisting to SQLite."""
        evidence_path = None
        if evidence_image is not None:
            evidence_path = self.save_evidence_image(evidence_image, prefix=event_category.lower().replace(" ", "_"))

        incident_id = self.db.insert_incident(
            source_filename=source_filename,
            source_type=source_type,
            event_category=event_category,
            confidence=confidence,
            rule_status=rule_status,
            timestamp_seconds=timestamp_seconds,
            evidence_path=evidence_path,
            reviewer_notes=reviewer_notes
        )
        logger.info(f"Logged {event_category} incident #{incident_id} from {source_filename}")
        return incident_id

    def list_incidents(
        self,
        status: Optional[str] = None,
        category: Optional[str] = None,
        source_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return self.db.get_all_incidents(status_filter=status, category_filter=category, source_type_filter=source_type)

    def update_review(self, incident_id: int, status: str, notes: str = "") -> bool:
        return self.db.update_review_status(incident_id, status, notes)

    def delete(self, incident_id: int) -> bool:
        return self.db.delete_incident(incident_id)

    def get_summary_metrics(self) -> Dict[str, Any]:
        return self.db.get_statistics()
