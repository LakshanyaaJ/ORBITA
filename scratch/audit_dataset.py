import glob
import json
import os

train_txts = sorted(glob.glob('datasets/orbita/labels/train/*.txt'))
val_txts = sorted(glob.glob('datasets/orbita/labels/val/*.txt'))
test_txts = sorted(glob.glob('datasets/orbita/labels/test/*.txt'))

print(f"Train: {len(train_txts)}, Val: {len(val_txts)}, Test: {len(test_txts)}")
for split, flist in [('train', train_txts), ('val', val_txts), ('test', test_txts)]:
    for tf in flist:
        meta_f = tf.replace('.txt', '.meta.json')
        if os.path.exists(meta_f):
            with open(meta_f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            cnames = [b.get('class_name', '') for b in data.get('boxes', [])]
            status = data.get('status', 'unknown')
            print(f"{split} - {os.path.basename(tf)}: {cnames} status={status}")
