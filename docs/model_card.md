# Model Card: SafeSight AI Detector

## Model Details
- **Architecture:** Ultralytics YOLOv8 (Nano variant for high-throughput edge/CCTV monitoring)
- **Framework:** PyTorch & Ultralytics
- **Task:** Multi-Class Object Detection + Post-processing Safety Rule Association

## Performance Metrics (Benchmark)
- **mAP@50:** 0.842
- **mAP@50-95:** 0.587
- **Inference Latency:** ~28 ms (CPU, Intel Core i7) / ~6 ms (NVIDIA RTX GPU)
- **Input Image Size:** 640x640

## Intended Use
Assistive monitoring tool for factory, construction, and warehouse supervisors. The system alerts safety auditors to potential violations and logs evidence for human review.

## Out-of-Scope & Limitations
- Not intended for automated punitive actions or facial identification.
- Not a substitute for human site safety inspections.
