import argparse
from ultralytics import YOLO

def evaluate_model(model_path: str, data_yaml: str, device: str = "cpu"):
    """Evaluates checkpoint on held-out dataset and outputs metrics."""
    print(f"Loading {model_path} for evaluation on {data_yaml}...")
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, device=device)
    print("=== Validation Results ===")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to checkpoint")
    parser.add_argument("--data", type=str, default="data/construction-ppe.yaml", help="Path to dataset YAML")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()
    evaluate_model(args.model, args.data, args.device)
