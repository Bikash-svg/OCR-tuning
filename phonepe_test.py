import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)

targets = [
    'screenshots/phonePe-70.jpg',    # control (works)
    'screenshots/phonePe-700.jpg',   # fails
    'screenshots/phonePe-7000.jpg',  # fails
    'screenshots/phonePe-7576.jpg',  # fails
]

for path in targets:
    img = Image.open(path)
    w, h = img.size
    print(f"\n{'=' * 70}")
    print(f"{path}  (image size {w}×{h})")
    print('=' * 70)

    page = ocr.predict(path)[0]
    for t, c, box in zip(page['rec_texts'], page['rec_scores'], page['rec_boxes']):
        b = np.array(box).tolist()
        print(f"  {t!r:<25}  conf={c:.2f}  box={b}")