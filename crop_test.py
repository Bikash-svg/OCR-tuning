from PIL import Image, ImageOps
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)

targets = [
    'screenshots/navi-3.jpg',
    'screenshots/navi-4.jpg',
    'screenshots/navi-5.jpg',
    'screenshots/navi-10.jpg',   # control (worked before)
]

for img_path in targets:
    img = Image.open(img_path).convert('RGB')
    w, h = img.size

    # Wider band — Navi amount sits lower than GPay's
    crop = img.crop((int(w*0.05), int(h*0.20), int(w*0.95), int(h*0.80)))
    crop = ImageOps.autocontrast(crop.convert('L'), cutoff=2)
    crop = crop.resize((crop.width*2, crop.height*2), Image.LANCZOS)

    out = f"/tmp/{img_path.split('/')[-1]}_crop2.png"
    crop.save(out)

    print(f"\n=== {img_path} CROPPED (wider band) ===")
    for page in ocr.predict(out):
        for t, c in zip(page.get('rec_texts', []), page.get('rec_scores', [])):
            print(f"  '{t}'  ({c:.2f})")