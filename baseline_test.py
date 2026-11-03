import glob, json
from paddleocr import PaddleOCR

# enable_mkldnn=False works around the PaddlePaddle 3.3.0 oneDNN PIR regression
ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)

rows = []
for img_path in sorted(glob.glob('screenshots/*')):
    if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    print(f"\n=== {img_path} ===")
    result = ocr.predict(img_path)
    lines = []
    if result:
        for page in result:
            texts = page.get('rec_texts', [])
            scores = page.get('rec_scores', [])
            for text, conf in zip(texts, scores):
                print(f"  '{text}'  (conf={conf:.2f})")
                lines.append({"text": text, "confidence": round(float(conf), 3)})
    rows.append({"image": img_path, "lines": lines})

with open('baseline_output.json', 'w', encoding='utf-8') as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print(f"\nSaved baseline_output.json ({len(rows)} images)")
