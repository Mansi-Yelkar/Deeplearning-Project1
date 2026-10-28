import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml

def train_safesight_model(
    data_yaml: str,
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    pretrained: str = "yolov8n.pt",
    device: str = "cpu",
    output_dir: str = "runs/train"
):
    """Fine-tunes lightweight YOLO detector on Workplace PPE dataset per Section 4.3."""
    print(f"=== SafeSight AI Fine-Tuning Pipeline ===")
    print(f"Dataset config: {data_yaml}")
    print(f"Pretrained checkpoint: {pretrained}")
    print(f"Epochs: {epochs} | Batch: {batch_size} | Resolution: {img_size}")
    
    model = YOLO(pretrained)
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=output_dir,
        name="safesight_ppe",
        save=True
    )
    print("Fine-tuning completed successfully.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SafeSight AI PPE Detection Model")
    parser.add_argument("--data", type=str, default="data/construction-ppe.yaml", help="Path to data YAML")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--pretrained", type=str, default="yolov8n.pt", help="Pretrained weights")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda:0)")
    args = parser.parse_args()

    train_safesight_model(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        pretrained=args.pretrained,
        device=args.device
    )
