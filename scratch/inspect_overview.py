import cv2
from pathlib import Path

for img_name in [
    "overview_20260908_135006.jpg",
    "overview_20260905_145858.jpg",
    "20260908_135006_f60.jpg",
    "desk_annotated_check.jpg",
    "sub_desk_pen_watch.jpg",
]:
    p = Path(img_name)
    if p.exists():
        im = cv2.imread(str(p))
        if im is not None:
            print(f"{p.name:<30}: shape = {im.shape}")
