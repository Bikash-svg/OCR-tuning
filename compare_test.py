import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)

paths = ['screenshots/gpay-2.jpg', 'screenshots/gpay-2-2.jpg']

for path in paths:
    print(f"\n=== {path} ===")
    img = Image.open(path)
    print(f"image size: {img.size}")
    page = ocr.predict(path)[0]

    amount_box = None
    for t, c, box in zip(page['rec_texts'], page['rec_scores'], page['rec_boxes']):
        b = np.array(box).tolist()
        print(f"  {t!r}  ({c:.2f})  box={b}")
        # capture the pure-digit line (the amount)
        if t.strip().isdigit() or (t.strip().startswith('₹') and any(ch.isdigit() for ch in t)):
            if amount_box is None:
                amount_box = b

    if amount_box:
        x1, y1, x2, y2 = amount_box
        pad = 20
        crop = img.crop((max(0, x1-pad), max(0, y1-pad), min(img.width, x2+pad), min(img.height, y2+pad)))
        # upscale for VISUAL inspection only (not fed back to OCR)
        big = crop.resize((crop.width*4, crop.height*4), Image.LANCZOS)
        out = f"/tmp/{path.split('/')[-1]}_amount.png"
        big.save(out)
        print(f"  → amount crop saved to {out}")

        # also OCR the raw (non-upscaled) crop
        raw_out = f"/tmp/{path.split('/')[-1]}_amount_raw.png"
        crop.save(raw_out)
        print(f"  → raw crop OCR:")
        for p in ocr.predict(raw_out):
            for t, c in zip(p['rec_texts'], p['rec_scores']):
                print(f"      {t!r}  ({c:.2f})")