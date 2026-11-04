"""
Hybrid pipeline: colour primary, greyscale fallback.

For each screenshot:
  1. Run OCR on the colour image
  2. Extract the topmost numeric amount candidate
  3. If it has ₹ OR has confidence >= 0.90 (fix will prepend ₹) → accept
  4. Otherwise, run OCR on a greyscale conversion and try again
  5. If both fail → return None (needs manual entry)
"""

import glob, json, re, os
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

BW_DIR = 'screenshots_bw'
os.makedirs(BW_DIR, exist_ok=True)

ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)

def is_redacted(t):
    return bool(re.search(r'x{3,}', t, re.IGNORECASE))

def fix_amount(text, confidence):
    clean = text.strip()
    if not re.fullmatch(r'\d[\d,]*(?:\.\d{1,2})?', clean):
        return text
    return f'₹{clean}' if confidence >= 0.90 else text

def extract_amount(lines):
    """Return (raw, fixed, conf, accepted) for the topmost numeric candidate."""
    candidates = [
        ln for ln in lines
        if re.fullmatch(r'₹?\d[\d,]*(?:\.\d{1,2})?', ln['text'].strip())
        and len(ln['text']) <= 12
    ]
    if not candidates:
        return (None, None, None, False)
    top = min(candidates, key=lambda x: x['box'][1])
    fixed = fix_amount(top['text'], top['confidence'])
    # accepted = has ₹ after fix AND confidence above garbage threshold
    accepted = fixed.startswith('₹') and top['confidence'] >= 0.40
    return (top['text'], fixed, top['confidence'], accepted)

def ocr_lines(path):
    result = ocr.predict(path)
    lines = []
    if result:
        page = result[0]
        for t, c, box in zip(page['rec_texts'], page['rec_scores'], page['rec_boxes']):
            if is_redacted(t):
                continue
            b = np.array(box).tolist()
            lines.append({'text': t, 'confidence': round(float(c), 3), 'box': b})
    return lines

rows = []
print(f"\n{'=' * 82}")
print(f"{'image':<22} {'COLOUR':<20} {'B&W':<20} {'FINAL':<12} {'via'}")
print(f"{'=' * 82}")

stats = {'color': 0, 'bw_fallback': 0, 'both_fail': 0}

for img_path in sorted(glob.glob('screenshots/*')):
    if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    if 'test_dummy' in img_path:
        continue

    # --- 1. Colour ---
    color_lines = ocr_lines(img_path)
    c_raw, c_fixed, c_conf, c_ok = extract_amount(color_lines)

    # --- 2. B&W fallback only if colour failed ---
    bw_raw = bw_fixed = None
    bw_conf = None
    if not c_ok:
        img = Image.open(img_path).convert('L')
        bw_path = os.path.join(BW_DIR, os.path.basename(img_path).rsplit('.', 1)[0] + '_bw.png')
        img.save(bw_path)
        bw_lines = ocr_lines(bw_path)
        bw_raw, bw_fixed, bw_conf, bw_ok = extract_amount(bw_lines)
    else:
        bw_ok = False

    # --- 3. Decide final ---
    if c_ok:
        final = c_fixed
        via = 'colour'
        stats['color'] += 1
    elif bw_ok:
        final = bw_fixed
        via = 'B&W fallback'
        stats['bw_fallback'] += 1
    else:
        final = 'FAILED'
        via = 'manual entry'
        stats['both_fail'] += 1

    rows.append({
        'image': img_path,
        'colour': {'raw': c_raw, 'fixed': c_fixed, 'conf': c_conf, 'ok': c_ok},
        'bw':     {'raw': bw_raw, 'fixed': bw_fixed, 'conf': bw_conf, 'ok': bw_ok},
        'final':  final, 'via': via,
    })

    c_str = f"{c_raw}->{c_fixed}" if c_raw else "NONE"
    bw_str = f"{bw_raw}->{bw_fixed}" if bw_raw else "(skipped)" if c_ok else "NONE"
    print(f"{os.path.basename(img_path):<22} {c_str:<20} {bw_str:<20} {final:<12} {via}")

print(f"\n{'=' * 82}")
print(f"Colour succeeded:       {stats['color']}")
print(f"B&W fallback rescued:   {stats['bw_fallback']}")
print(f"Both failed:            {stats['both_fail']}")
total = sum(stats.values())
print(f"Total:                  {stats['color'] + stats['bw_fallback']} / {total} successful")
print(f"{'=' * 82}")

with open('baseline_hybrid_output.json', 'w', encoding='utf-8') as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print("\nSaved baseline_hybrid_output.json")