import cv2
from pathlib import Path

train_imgs = list(Path("datasets/orbita/v4/images/train").glob("*.*"))[:10]
print(f"Total train images: {len(list(Path('datasets/orbita/v4/images/train').glob('*.*')))}")
for p in train_imgs:
    img = cv2.imread(str(p))
    if img is not None:
        print(f"{p.name}: shape = {img.shape}")
