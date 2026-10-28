# SafeSight AI System Architecture

## Overview
SafeSight AI provides a modular deep-learning pipeline for industrial personal protective equipment (PPE) compliance monitoring and incident audit management.

```mermaid
graph TD
    A[User Upload / Input Stream] --> B[Media Processor]
    B -->|Frames / Images| C[Object Detection Engine - YOLO]
    C -->|Worker & PPE Bounding Boxes| D[Safety Rule Engine]
    D -->|Compliance & Violations| E[Incident Manager]
    E -->|Audit Trail & Evidence Frames| F[(SQLite Database)]
    F --> G[Interactive Streamlit Dashboard]
    F --> H[Incident Management & Human Review]
    F --> I[Analytics & CSV Export]
```

## Core Modules
1. **Media Processor (`src/video/video_processor.py` & `src/utils/file_validation.py`)**: Incrementally decodes video frames to prevent memory bloat, preserving exact timestamps.
2. **Detection Engine (`src/models/detector.py`)**: Wraps lightweight YOLO models with dataset-agnostic class normalization.
3. **Safety Rule Engine (`src/safety/ppe_rules.py` & `src/safety/zone_rules.py`)**: Spatial association associating detected helmets and vests with detected workers.
4. **Incident Database (`src/incidents/database.py`)**: SQLite storage recording incidents, review states (`pending`, `confirmed`, `dismissed`, `uncertain`), reviewer notes, and evidence snapshots.
5. **Dashboard & UI (`app.py`, `pages/`)**: Multi-page Streamlit application for end-to-end audit, review, and analytics.
