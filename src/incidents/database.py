import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

class IncidentDatabase:
    """SQLite-backed persistent store for safety violation incidents."""
    
    def __init__(self, db_path: str = "incidents.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes incidents schema if not already present."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_filename TEXT NOT NULL,
                    source_type TEXT NOT NULL,          -- 'image' or 'video'
                    timestamp_seconds REAL,             -- video timestamp in seconds or NULL
                    event_category TEXT NOT NULL,       -- e.g., 'Missing Helmet', 'Missing Vest', 'Restricted Zone'
                    confidence REAL NOT NULL,
                    rule_status TEXT NOT NULL,          -- 'violation', 'compliant', 'uncertain'
                    review_status TEXT DEFAULT 'pending', -- 'pending', 'confirmed', 'dismissed', 'uncertain'
                    evidence_path TEXT,
                    reviewer_notes TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def insert_incident(
        self,
        source_filename: str,
        source_type: str,
        event_category: str,
        confidence: float,
        rule_status: str,
        timestamp_seconds: Optional[float] = None,
        evidence_path: Optional[str] = None,
        review_status: str = "pending",
        reviewer_notes: str = ""
    ) -> int:
        """Inserts a new incident record and returns its ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO incidents (
                    source_filename, source_type, timestamp_seconds,
                    event_category, confidence, rule_status,
                    review_status, evidence_path, reviewer_notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                source_filename,
                source_type,
                timestamp_seconds,
                event_category,
                round(float(confidence), 4),
                rule_status,
                review_status,
                evidence_path,
                reviewer_notes,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()
            return cursor.lastrowid

    def get_all_incidents(
        self,
        status_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        source_type_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves incidents with optional filtering."""
        query = "SELECT * FROM incidents WHERE 1=1"
        params: List[Any] = []
        
        if status_filter and status_filter.lower() != "all":
            query += " AND review_status = ?"
            params.append(status_filter.lower())
            
        if category_filter and category_filter.lower() != "all":
            query += " AND event_category = ?"
            params.append(category_filter)
            
        if source_type_filter and source_type_filter.lower() != "all":
            query += " AND source_type = ?"
            params.append(source_type_filter.lower())
            
        query += " ORDER BY id DESC"
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def update_review_status(self, incident_id: int, review_status: str, reviewer_notes: str = "") -> bool:
        """Updates review status ('pending', 'confirmed', 'dismissed', 'uncertain') and optional reviewer notes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE incidents
                SET review_status = ?, reviewer_notes = ?
                WHERE id = ?
            """, (review_status.lower(), reviewer_notes, incident_id))
            conn.commit()
            return cursor.rowcount > 0

    def delete_incident(self, incident_id: int) -> bool:
        """Deletes an incident by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM incidents WHERE id = ?", (incident_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_all(self) -> None:
        """Clears all records for testing purposes."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM incidents")
            conn.commit()

    def get_statistics(self) -> Dict[str, Any]:
        """Calculates aggregated metrics for dashboard analytics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM incidents")
            total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM incidents WHERE review_status = 'pending'")
            pending = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM incidents WHERE review_status = 'confirmed'")
            confirmed = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM incidents WHERE review_status = 'dismissed'")
            dismissed = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM incidents WHERE review_status = 'uncertain'")
            uncertain = cursor.fetchone()[0]

            cursor.execute("SELECT event_category, COUNT(*) FROM incidents GROUP BY event_category")
            category_counts = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute("SELECT source_type, COUNT(*) FROM incidents GROUP BY source_type")
            source_counts = {row[0]: row[1] for row in cursor.fetchall()}

            return {
                "total_incidents": total,
                "pending_review": pending,
                "confirmed": confirmed,
                "dismissed": dismissed,
                "uncertain": uncertain,
                "by_category": category_counts,
                "by_source": source_counts
            }
