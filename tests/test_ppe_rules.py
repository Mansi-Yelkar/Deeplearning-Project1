import pytest
from src.models.model_loader import Detection
from src.safety.ppe_rules import PPERuleEngine, calculate_iou, calculate_containment

def test_iou_calculation():
    boxA = [0, 0, 10, 10]
    boxB = [0, 0, 10, 10]
    assert calculate_iou(boxA, boxB) == 1.0

    boxC = [20, 20, 30, 30]
    assert calculate_iou(boxA, boxC) == 0.0

def test_containment():
    inner = [2, 2, 4, 4]
    outer = [0, 0, 10, 10]
    assert calculate_containment(inner, outer) == 1.0

def test_ppe_rule_compliant_worker():
    engine = PPERuleEngine(require_helmet=True, require_vest=True)
    
    # Worker from [100, 100, 200, 400]
    worker = Detection(class_id=0, class_name="person", confidence=0.92, box=[100, 100, 200, 400])
    # Helmet on upper head [110, 95, 190, 150]
    helmet = Detection(class_id=1, class_name="helmet", confidence=0.89, box=[110, 95, 190, 150])
    # Vest on torso [110, 160, 190, 320]
    vest = Detection(class_id=2, class_name="vest", confidence=0.85, box=[110, 160, 190, 320])

    assessment = engine.assess_worker(worker, [worker, helmet, vest])
    assert assessment.has_helmet is True
    assert assessment.has_vest is True
    assert assessment.is_violation is False
    assert len(assessment.violation_reasons) == 0

def test_ppe_rule_missing_helmet():
    engine = PPERuleEngine(require_helmet=True, require_vest=True)
    worker = Detection(class_id=0, class_name="person", confidence=0.90, box=[100, 100, 200, 400])
    vest = Detection(class_id=2, class_name="vest", confidence=0.85, box=[110, 160, 190, 320])

    assessment = engine.assess_worker(worker, [worker, vest])
    assert assessment.has_helmet is False
    assert assessment.is_violation is True
    assert "Missing Helmet" in assessment.violation_reasons

def test_ppe_rule_missing_vest():
    engine = PPERuleEngine(require_helmet=True, require_vest=True)
    worker = Detection(class_id=0, class_name="person", confidence=0.90, box=[100, 100, 200, 400])
    helmet = Detection(class_id=1, class_name="helmet", confidence=0.89, box=[110, 95, 190, 150])

    assessment = engine.assess_worker(worker, [worker, helmet])
    assert assessment.has_vest is False
    assert assessment.is_violation is True
    assert "Missing Safety Vest" in assessment.violation_reasons
