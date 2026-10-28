# Dataset Card: Construction-PPE & Industrial Safety Dataset

## 1. Summary
The Construction-PPE dataset is utilized for training and evaluating object detection models targeting worker safety equipment in industrial, construction, and manufacturing environments.

- **Primary Source:** [Ultralytics Construction-PPE Dataset](https://docs.ultralytics.com/datasets/detect/construction-ppe/)
- **Alternative / Supplementary:** [Roboflow Hard Hat Workers Dataset](https://public.roboflow.com/object-detection/hard-hat-workers/)
- **Task:** 2D Bounding Box Object Detection

## 2. Supported Classes
- `person` / `worker`: Individual human worker
- `helmet` / `hard-hat`: Approved protective safety headgear
- `vest` / `safety-vest`: High-visibility safety vest
- `head`: Worker head without helmet (where annotated)
- `no-helmet`: Worker without hard hat (in explicit violation datasets)
- `no-vest`: Worker without reflective vest

## 3. Recommended Acquisition
To download and prepare the dataset for training:
```bash
python scripts/train.py --data data/construction-ppe.yaml --epochs 50 --batch 16
```
Ultralytics automatically downloads and structures the benchmark dataset when pointed to the standard configuration.

## 4. Limitations & Edge Cases
- **Class Imbalance:** In operational sites, compliant workers often outnumber non-compliant workers.
- **Occlusions:** Scaffolding, machinery, and extreme camera angles may partially obscure hard hats or reflective vests.
- **Distance / Scale:** Small workers in distant CCTV views (<30 pixels) have lower recall.
