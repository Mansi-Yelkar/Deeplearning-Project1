# 🛡️ SafeSight AI — Deep Learning-Based Workplace Safety Monitoring System

SafeSight AI is an intelligent workplace safety and PPE (Personal Protective Equipment) compliance monitoring system designed for industrial facilities, construction sites, and warehouses.

---

## 🌟 Key Features
- **Object Detection Pipeline:** Detects workers, helmets, and high-visibility safety vests with Ultralytics YOLO.
- **Decoupled Safety Rule Engine:** Assesses spatial relationships to confirm PPE compliance, flagging missing helmets or safety vests.
- **Restricted-Zone Monitoring:** Optional polygonal hazard zone incursion detection.
- **Incremental Video Processing:** Streamlined frame-by-frame analysis with exact timestamp preservation.
- **Incident Audit Trail:** Persistent SQLite database tracking review statuses (`pending`, `confirmed`, `dismissed`, `uncertain`) and reviewer notes.
- **Interactive Industrial UI:** Multi-page Streamlit dashboard featuring live analytics, review workflows, and CSV export.

---

## 🏗️ Project Architecture
```text
SafeSight-AI/
├── app.py                      # Main Dashboard application
├── requirements.txt            # Pinned dependencies
├── config/settings.yaml        # System settings
├── pages/                      # Streamlit navigation pages
│   ├── 1_Image_Analysis.py
│   ├── 2_Video_Analysis.py
│   ├── 3_Incident_Management.py
│   ├── 4_Analytics.py
│   └── 5_Model_Evaluation.py
├── src/                        # Modular engine implementation
│   ├── models/detector.py      # YOLO detector wrapper
│   ├── safety/ppe_rules.py     # Worker-to-PPE association logic
│   ├── safety/zone_rules.py    # Restricted zone incursion logic
│   ├── video/video_processor.py# Memory-safe incremental video reader
│   ├── incidents/database.py   # SQLite persistence layer
│   └── utils/file_validation.py# File security & size validator
├── scripts/                    # Training and offline evaluation scripts
├── tests/                      # Automated unit test suite
└── docs/                       # Architecture, dataset, and model cards
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone repository and install requirements:
```bash
pip install -r requirements.txt
```

### 2. Run the Application
Launch the interactive Streamlit dashboard:
```bash
streamlit run app.py
```

### 3. Run Automated Tests
Execute the pytest suite:
```bash
python -m pytest tests/
```

### 4. Training / Fine-Tuning
```bash
python scripts/train.py --data data/construction-ppe.yaml --epochs 50 --batch 16
```

---

## 📋 Ethical and Responsible AI Statement
SafeSight AI functions as an assistive tool to support qualified workplace safety officers. It does not perform facial recognition, does not track individual biometric identity, and mandates human verification before safety infractions are confirmed.
