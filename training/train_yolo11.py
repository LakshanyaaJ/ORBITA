import os
import sys
import yaml
import shutil
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_yolo11")

def train_yolo11(config_path: str = "training/configs/yolo11_config.yaml"):
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
        
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)
        
    logger.info(f"Loaded training configuration: {cfg}")
    
    base_model_name = cfg.get("model", "yolo11n.pt")
    logger.info(f"Initializing base model: {base_model_name}")
    
    try:
        model = YOLO(base_model_name)
    except Exception as e:
        logger.warning(f"Could not load {base_model_name} directly ({e}). Falling back to yolo8n base for initialization if needed.")
        model = YOLO("yolov8n.pt")
        
    data_path = cfg.get("data", "datasets/orbita/v3/data.yaml")
    epochs = cfg.get("epochs", 15)
    imgsz = cfg.get("imgsz", 640)
    batch = cfg.get("batch", 8)
    device = cfg.get("device", "cpu")
    project = cfg.get("project", "runs/orbita")
    name = cfg.get("name", "yolo11_experiment")
    
    logger.info(f"Starting training on data {data_path} for {epochs} epochs...")
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True
    )
    
    logger.info("Training complete.")
    
    # Save the trained model checkpoint to models/orbita_yolo11.pt
    best_weights_path = Path(project) / name / "weights" / "best.pt"
    dest_path = Path("models/orbita_yolo11.pt")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if best_weights_path.exists():
        shutil.copyfile(best_weights_path, dest_path)
        logger.info(f"Saved upgraded YOLO11 model to {dest_path}")
    else:
        # Fallback to current model save
        model.save(str(dest_path))
        logger.info(f"Saved YOLO11 model directly to {dest_path}")
        
    return str(dest_path)

if __name__ == "__main__":
    train_yolo11()
