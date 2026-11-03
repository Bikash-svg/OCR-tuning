import glob, json
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

rows = []
for img_path in sorted(glob.glob('screenshots/*')):
    if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    print(f"\n=== {img_path} ===")
    result = ocr.ocr(img_path, cls=True)
    lines = []
    if result and result[0]:
        for box, (text, confidence) in result[0]:
            print(f"  '{text}'  (conf={confidence:.2f})")
            lines.append({"text": text, "confidence": round(confidence, 3)})
    rows.append({"image": img_path, "lines": lines})

with open('baseline_output.json', 'w', encoding='utf-8') as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print(f"\nSaved baseline_output.json ({len(rows)} images)")
