import glob, json, re, csv
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)

def is_redacted(text):
    return bool(re.search(r'x{3,}', text, re.IGNORECASE))

def fix_amount(text, confidence):
    """Prepend ₹ to high-confidence pure-digit amounts (model dropped the symbol)."""
    clean = text.strip()
    if not re.fullmatch(r'\d[\d,]*(?:\.\d{1,2})?', clean):
        return text
    if confidence >= 0.90:
        return f'₹{clean}'
    return text

rows, scores = [], []

for img_path in sorted(glob.glob('screenshots/*')):
    if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    if 'test_dummy' in img_path:
        continue

    print(f"\n=== {img_path} ===")
    result = ocr.predict(img_path)
    lines = []
    if result:
        page = result[0]
        for t, c, box in zip(page['rec_texts'], page['rec_scores'], page['rec_boxes']):
            if is_redacted(t):
                continue
            b = np.array(box).tolist()
            y_center = (b[1] + b[3]) / 2
            fixed = fix_amount(t, c)
            flag = '  ← FIXED' if fixed != t else ''
            print(f"  {t!r}  ({c:.2f})  y={y_center:.0f}{flag}")
            lines.append({"text": t, "fixed": fixed,
                          "confidence": round(float(c), 3), "box": b})
    rows.append({"image": img_path, "lines": lines})

    candidates = []
    for ln in lines:
        t = ln['text'].strip()
        if re.fullmatch(r'₹?\d[\d,]*(?:\.\d{1,2})?', t) and len(t) <= 12:
            candidates.append(ln)
    if candidates:
        top = min(candidates, key=lambda x: x['box'][1])
        scores.append({"image": img_path, "raw": top['text'],
                       "fixed": top['fixed'], "conf": top['confidence']})
    else:
        scores.append({"image": img_path, "raw": None,
                       "fixed": None, "conf": None})

with open('baseline_output.json', 'w', encoding='utf-8') as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 76)
print("SUMMARY — topmost amount, raw vs after fix")
print("=" * 76)
print(f"{'image':<22} {'raw':<8} {'fixed':<8} {'conf':<6} {'₹ before':<10} {'₹ after'}")
before = after = 0
for s in scores:
    name = s['image'].replace('screenshots/', '')
    raw = s['raw'] or 'NONE'
    fixed = s['fixed'] or 'NONE'
    conf = f"{s['conf']:.2f}" if s['conf'] is not None else '-'
    rb = 'YES' if (s['raw'] and s['raw'].startswith('₹')) else 'NO'
    ra = 'YES' if (s['fixed'] and s['fixed'].startswith('₹')) else 'NO'
    if rb == 'YES': before += 1
    if ra == 'YES': after += 1
    print(f"{name:<22} {raw:<8} {fixed:<8} {conf:<6} {rb:<10} {ra}")

print(f"\n₹ present BEFORE fix: {before} / {len(scores)}")
print(f"₹ present AFTER  fix: {after} / {len(scores)}")

with open('baseline_scores.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['image', 'true_amount', 'ocr_raw', 'ocr_fixed',
                'match_raw', 'match_fixed', 'error_type', 'notes'])
    for s in scores:
        name = s['image'].replace('screenshots/', '')
        w.writerow([name, '', s['raw'] or '', s['fixed'] or '', '', '', '', ''])

print("\nSaved baseline_output.json")
print("Saved baseline_scores.csv")