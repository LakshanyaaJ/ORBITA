import cv2
from core_ai.app.config import load_config
from core_ai.perception.object_detector import ObjectDetector

cfg = load_config()
detector = ObjectDetector(cfg.detection)

cap = cv2.VideoCapture("vdata/20260908_135006.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 1814)
ret, frame = cap.read()
print("Read frame 1814:", ret, frame.shape if frame is not None else None)

if frame is not None:
    # Test YOLO detections directly
    yolo_dets = detector._detect_yolo(frame, 0.0)
    print("YOLO dets:", [(d.class_name, d.confidence, d.bbox) for d in yolo_dets])
    
    # Test Chroma fallback detections directly
    chroma_dets = detector._detect_chroma(frame, 0.0)
    print("Chroma dets:", [(d.class_name, d.confidence, d.bbox) for d in chroma_dets])
    
    # Test full detect()
    full_dets = detector.detect(frame, 0.0)
    print("Full detect:", [(d.class_name, d.confidence, d.bbox) for d in full_dets])
    print("Grouped detections:", detector.get_grouped_detections())

cap.release()
