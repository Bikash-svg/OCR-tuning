"""
Black & white variant of baseline_test.py.

Converts every screenshot to greyscale, saves to screenshots_bw/,
runs OCR on the greyscale version, applies the same confidence-gated
rupee fix, and prints a side-by-side comparison with the raw (color) result.

The purpose: test whether color is the factor causing Navi/GPay ₹ drop-outs.
If a screenshot that failed in color suddenly reads ₹ in greyscale → color
was a factor. If it still fails → animation/render is the factor.
"""

import glob, json, re, csv, os
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BW_DIR = 'screenshots_bw'
os.makedirs(BW_DIR, exist_ok=True)

ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
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

def extract_top_amount(lines):
    """Pick the topmost short numeric candidate from a list of OCR lines."""
    candidates = [
        ln for ln in lines
        if re.fullmatch(r'₹?\d[\d,]*(?:\.\d{1,2})?', ln['text'].strip())
        and len(ln['text']) <= 12
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda x: x['box'][1])

# ------------------------------------------------------------------
# Load the color (original) baseline result if it exists
# ------------------------------------------------------------------
color_results = {}
if os.path.exists('baseline_output.json'):
    with open('baseline_output.json', encoding='utf-8') as f:
        for row in json.load(f):
            top = extract_top_amount(row['lines'])
            color_results[row['image']] = {
                'raw': top['text'] if top else None,
                'conf': top['confidence'] if top else None,
            }

# ------------------------------------------------------------------
# Process each screenshot in greyscale
# ------------------------------------------------------------------
bw_rows = []
print(f"\n{'=' * 90}")
print(f"{'image':<22} {'COLOR':<16} {'B&W':<16} {'B&W conf':<10} {'verdict'}")
print(f"{'=' * 90}")

wins = 0
losses = 0
ties = 0

for img_path in sorted(glob.glob('screenshots/*')):
    if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    if 'test_dummy' in img_path:
        continue

    # Convert to greyscale and save
    img = Image.open(img_path).convert('L')
    bw_path = os.path.join(BW_DIR, os.path.basename(img_path).rsplit('.', 1)[0] + '_bw.png')
    img.save(bw_path)

    # OCR the greyscale version
    result = ocr.predict(bw_path)
    lines = []
    if result:
        page = result[0]
        for t, c, box in zip(page['rec_texts'], page['rec_scores'], page['rec_boxes']):
            if is_redacted(t):
                continue
            b = np.array(box).tolist()
            lines.append({'text': t, 'confidence': round(float(c), 3), 'box': b})
    bw_rows.append({'image': img_path, 'bw_path': bw_path, 'lines': lines})

    # Extract topmost amount
    top_bw = extract_top_amount(lines)
    bw_raw = top_bw['text'] if top_bw else 'NONE'
    bw_conf = top_bw['confidence'] if top_bw else None
    bw_fixed = fix_amount(bw_raw, bw_conf) if top_bw else 'NONE'

    # Compare with color
    color = color_results.get(img_path, {'raw': None, 'conf': None})
    color_raw = color['raw'] or 'NONE'
    color_fixed = fix_amount(color_raw, color['conf']) if color['conf'] is not None else 'NONE'

    # Verdict: did greyscale help or hurt?
    if color_fixed.startswith('₹') and bw_fixed.startswith('₹'):
        verdict = 'BOTH OK'
        ties += 1
    elif color_fixed.startswith('₹') and not bw_fixed.startswith('₹'):
        verdict = 'B&W LOST IT'
        losses += 1
    elif not color_fixed.startswith('₹') and bw_fixed.startswith('₹'):
        verdict = 'B&W FIXED IT'
        wins += 1
    else:
        verdict = 'BOTH FAILED'
        ties += 1

    color_str = f"{color_raw}->{color_fixed}"
    bw_str = f"{bw_raw}->{bw_fixed}"
    conf_str = f"{bw_conf:.2f}" if bw_conf is not None else '-'

    print(f"{os.path.basename(img_path):<22} {color_str:<16} {bw_str:<16} {conf_str:<10} {verdict}")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print(f"\n{'=' * 90}")
print(f"B&W FIXED:  {wins}")
print(f"B&W LOST:   {losses}")
print(f"NO CHANGE:  {ties}")
print(f"{'=' * 90}")

# Save the greyscale OCR output for reference
with open('baseline_bw_output.json', 'w', encoding='utf-8') as f:
    json.dump(bw_rows, f, ensure_ascii=False, indent=2)
print(f"\nSaved baseline_bw_output.json ({len(bw_rows)} images)")
print(f"Greyscale images saved to {BW_DIR}/")

# ------------------------------------------------------------------
# Interpretation guide
# ------------------------------------------------------------------
print("""
INTERPRETATION:
  B&W FIXED IT  → colour was a factor for that image; add greyscale preprocessing
  B&W LOST IT   → greyscale hurts; don't convert these
  BOTH OK       → already correct in colour, greyscale keeps it correct
  BOTH FAILED   → neither colour nor greyscale works; it's an animation/render issue
""")