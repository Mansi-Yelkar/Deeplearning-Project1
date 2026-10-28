import pytest
import os
from src.incidents.database import IncidentDatabase
from src.incidents.incident_manager import IncidentManager

@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_incidents.db"
    return str(db_file)

def test_insert_and_retrieve_incident(temp_db):
    db = IncidentDatabase(temp_db)
    inc_id = db.insert_incident(
        source_filename="test_site.jpg",
        source_type="image",
        event_category="Missing Helmet",
        confidence=0.88,
        rule_status="violation"
    )
    assert inc_id == 1

    all_records = db.get_all_incidents()
    assert len(all_records) == 1
    assert all_records[0]["event_category"] == "Missing Helmet"
    assert all_records[0]["review_status"] == "pending"

def test_update_review_status(temp_db):
    db = IncidentDatabase(temp_db)
    inc_id = db.insert_incident(
        source_filename="site.mp4",
        source_type="video",
        event_category="Missing Safety Vest",
        confidence=0.82,
        rule_status="violation"
    )
    success = db.update_review_status(inc_id, "confirmed", "Verified by Site Safety Officer")
    assert success is True

    updated = db.get_all_incidents()[0]
    assert updated["review_status"] == "confirmed"
    assert updated["reviewer_notes"] == "Verified by Site Safety Officer"

def test_incident_filtering(temp_db):
    db = IncidentDatabase(temp_db)
    db.insert_incident("1.jpg", "image", "Missing Helmet", 0.9, "violation", review_status="confirmed")
    db.insert_incident("2.jpg", "image", "Missing Safety Vest", 0.9, "violation", review_status="pending")

    pending_list = db.get_all_incidents(status_filter="pending")
    assert len(pending_list) == 1
    assert pending_list[0]["event_category"] == "Missing Safety Vest"
