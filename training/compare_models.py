import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.validate import evaluate_model

def compare_yolo_models(
    v8_path: str = "models/orbita_yolov8_backup.pt",
    v11_path: str = "models/orbita_yolo11.pt",
    data_yaml: str = "datasets/orbita/v3/data.yaml"
):
    print("==================================================")
    print("         ORBITA MODEL COMPARISON REPORT           ")
    print("==================================================")
    
    if not Path(v8_path).exists():
        v8_path = "models/orbita_yolo_detector_v3.pt"
        
    if not Path(v11_path).exists():
        v11_path = "models/orbita_yolo_detector_v4.pt"
        
    print(f"Evaluating YOLOv8 model: {v8_path}...")
    v8_metrics = evaluate_model(v8_path, data_yaml)
    
    print(f"Evaluating YOLO11 model: {v11_path}...")
    v11_metrics = evaluate_model(v11_path, data_yaml)
    
    header = f"{'Metric':<18} | {'YOLOv8':<12} | {'YOLO11':<12}"
    divider = "-" * len(header)
    
    v8_sz = f"{v8_metrics['model_size_mb']} MB"
    v11_sz = f"{v11_metrics['model_size_mb']} MB"
    
    print("\n" + divider)
    print(header)
    print(divider)
    
    print(f"{'mAP50':<18} | {v8_metrics['mAP50']:<12} | {v11_metrics['mAP50']:<12}")
    print(f"{'mAP50-95':<18} | {v8_metrics['mAP50-95']:<12} | {v11_metrics['mAP50-95']:<12}")
    print(f"{'Precision':<18} | {v8_metrics['precision']:<12} | {v11_metrics['precision']:<12}")
    print(f"{'Recall':<18} | {v8_metrics['recall']:<12} | {v11_metrics['recall']:<12}")
    print(f"{'FPS':<18} | {v8_metrics['fps']:<12} | {v11_metrics['fps']:<12}")
    print(f"{'Latency (ms)':<18} | {v8_metrics['latency_ms']:<12} | {v11_metrics['latency_ms']:<12}")
    print(f"{'Model Size (MB)':<18} | {v8_sz:<12} | {v11_sz:<12}")
    print(divider)
    
    report = {
        "yolov8": v8_metrics,
        "yolo11": v11_metrics
    }
    
    with open("models/yolo_comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print("\nSaved detailed comparison report to models/yolo_comparison_report.json")
    return report

if __name__ == "__main__":
    compare_yolo_models()
