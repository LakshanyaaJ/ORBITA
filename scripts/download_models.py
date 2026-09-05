"""Utility script to download or verify YOLOv8 model weights for ORBITA."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def download_models():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] Ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    models = ["yolov8n.pt", "yolov8n-pose.pt"]
    for model_name in models:
        target_path = MODELS_DIR / model_name
        if target_path.exists():
            print(f"[✓] {model_name} already exists at {target_path}")
        else:
            print(f"[*] Downloading {model_name}...")
            model = YOLO(model_name)
            # Copy or save to models dir
            downloaded = Path(model_name)
            if downloaded.exists() and downloaded != target_path:
                downloaded.rename(target_path)
            print(f"[✓] Saved {model_name} to {target_path}")

if __name__ == "__main__":
    download_models()
