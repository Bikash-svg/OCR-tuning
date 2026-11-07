"""
Hybrid pipeline: colour primary, greyscale fallback + field extraction.

Direction rules (priority order):
  1. self transfer
  2. paid to / payment to                 → debit
  3. "Paid in X seconds" (BHIM footer)    → debit
  4. "Paid" header in top 20% (BHIM)      → debit  ← catches bhim-3
  5. "{Name} Paid" EOL (FamPay)           → debit  (name >= 3 chars)
  6. sent to / send to / money sent       → debit
  7. received from / payment received     → credit
  8. debited from / credited to (weak)
  9. To UPI ID (debit) / From UPI ID (credit)
 10. standalone To: / From:

Counterparty priority (direction-aware):
  debit  → paid to / To UPI ID / "{Name} Paid" → name
  credit → received from / From UPI ID         → name
  fallback → banking name
  "screenshot" is always excluded.
"""

import glob, json, re, os, sys
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

BW_DIR = 'screenshots_bw'
os.makedirs(BW_DIR, exist_ok=True)
DEBUG = os.environ.get('DEBUG_OCR') == '1'

ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn=False)

# ---------------------------------------------------------------------------
# Amount
# ---------------------------------------------------------------------------
def is_redacted(t):
    return bool(re.search(r'x{3,}', t, re.IGNORECASE))

def fix_amount(text, confidence):
    clean = text.strip()
    if not re.fullmatch(r'\d[\d,]*(?:\.\d{1,2})?', clean):
        return text
    return f'₹{clean}' if confidence >= 0.90 else text

def extract_amount(lines, image_height=None):
    candidates = [
        ln for ln in lines
        if re.fullmatch(r'₹?\d[\d,]*(?:\.\d{1,2})?', ln['text'].strip())
        and len(ln['text']) <= 15
        and ln['confidence'] >= 0.40
    ]
    if not candidates:
        return (None, None, None, False)

    if image_height is not None:
        filtered = [c for c in candidates if c['box'][1] >= image_height * 0.08]
        if filtered:
            candidates = filtered

    tier_a = [c for c in candidates if c['text'].strip().startswith('₹') and ',' in c['text']]
    tier_b = [c for c in candidates if c['text'].strip().startswith('₹') and ',' not in c['text']]
    tier_c = [c for c in candidates if ',' in c['text'] and not c['text'].strip().startswith('₹')]
    if tier_a:   pool = tier_a
    elif tier_b: pool = tier_b
    elif tier_c: pool = tier_c
    else:        pool = candidates

    top = min(pool, key=lambda x: x['box'][1])
    fixed = fix_amount(top['text'], top['confidence'])
    accepted = fixed.startswith('₹') and top['confidence'] >= 0.40
    return (top['text'], fixed, top['confidence'], accepted)

# ---------------------------------------------------------------------------
# App detection
# ---------------------------------------------------------------------------
APP_NAME_MARKERS = [
    ('PhonePe',     ['phonepe', 'phone pe']),
    ('Navi',        ['navi upi', 'navi']),
    ('BHIM',        ['bhim']),
    ('FamPay',      ['fampay', 'fam pay', 'paid with fam', 'trio p']),
    ('GPay',        ['gpay', 'g pay', 'google pay']),
    ('Paytm',       ['paytm', 'pay tm']),
    ('Slice',       ['slice']),
    ('SuperMoney',  ['supermoney', 'super money', 'super.money']),
    ('BharatPe',    ['bharatpe', 'bharat pe']),
    ('Amazon Pay',  ['amazon pay', 'amazonpay']),
    ('Mobikwik',    ['mobikwik']),
    ('CRED',        ['cred upi']),
]

def detect_app(lines, img_path):
    text = ' '.join(ln['text'] for ln in lines).lower()
    for app, markers in APP_NAME_MARKERS:
        for m in markers:
            if m in text:
                return app
    fname = os.path.basename(img_path).lower()
    for key, app in [
        ('phonepe', 'PhonePe'), ('navi', 'Navi'), ('bhim', 'BHIM'),
        ('fampay', 'FamPay'), ('gpay', 'GPay'),
        ('paytm', 'Paytm'), ('slice', 'Slice'),
    ]:
        if key in fname:
            return app
    return 'unknown'

# ---------------------------------------------------------------------------
# Date / time
# ---------------------------------------------------------------------------
MONTHS = {
    'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
    'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
    'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9,
    'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12,
}

DATE_PATTERNS = [
    re.compile(
        r'\b(\d{1,2})(?:st|nd|rd|th)?\s+'
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*'
        r'[\s,]+\s*(\d{2}|\d{4})\b',
        re.IGNORECASE,
    ),
    re.compile(
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+'
        r'(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(\d{2}|\d{4})\b',
        re.IGNORECASE,
    ),
    re.compile(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b'),
]
TIME_PATTERNS = [
    re.compile(r'\b(\d{1,2}):(\d{2})\s*(am|pm)\b', re.IGNORECASE),
    re.compile(r'\b(\d{1,2}):(\d{2})\b'),
]

PAID_LINE_RE = re.compile(r'^(.+?)\s+Paid\s*$', re.IGNORECASE)

def _normalize_year(y):
    y_int = int(y)
    return y_int + 2000 if y_int < 100 else y_int

def _date_from(text):
    m = DATE_PATTERNS[0].search(text)
    if m:
        day, month_name, year = m.groups()
        month = MONTHS.get(month_name.lower())
        if month:
            return f"{_normalize_year(year):04d}-{month:02d}-{int(day):02d}"
    m = DATE_PATTERNS[1].search(text)
    if m:
        month_name, day, year = m.groups()
        month = MONTHS.get(month_name.lower())
        if month:
            return f"{_normalize_year(year):04d}-{month:02d}-{int(day):02d}"
    m = DATE_PATTERNS[2].search(text)
    if m:
        day, month, year = m.groups()
        return f"{_normalize_year(year):04d}-{int(month):02d}-{int(day):02d}"
    return None

def _time_from_line(lines, idx):
    text = lines[idx]['text']
    m = TIME_PATTERNS[0].search(text)
    if m:
        hour, minute, ampm = m.groups()
        h = int(hour) % 12
        if ampm.lower() == 'pm':
            h += 12
        return f"{h:02d}:{minute}"
    m = TIME_PATTERNS[1].search(text)
    if m:
        hour, minute = m.groups()
        base_y = lines[idx]['box'][1]
        ampm = None
        for j, other in enumerate(lines):
            if j == idx:
                continue
            if abs(other['box'][1] - base_y) < 100:
                am = re.match(r'^\s*(am|pm)\s*$', other['text'].strip(), re.IGNORECASE)
                if am:
                    ampm = am.group(1).lower()
                    break
        h = int(hour) % 12
        if ampm == 'pm':
            h += 12
        return f"{h:02d}:{minute}"
    return None

def find_date_time(lines):
    date_iso = time_24h = None
    for i, ln in enumerate(lines):
        if date_iso is None:
            date_iso = _date_from(ln['text'])
        if time_24h is None:
            time_24h = _time_from_line(lines, i)
        if date_iso and time_24h:
            break
    return date_iso, time_24h

# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------
def find_direction(lines, image_height=None):
    text = ' '.join(ln['text'].lower() for ln in lines)

    # 1. Self transfer
    if 'self transfer' in text:
        return 'self_transfer', 'self transfer'

    # 2. paid to / payment to → debit
    for kw in ['paid to', 'payment to']:
        if kw in text:
            return 'debit', kw

    # 3. BHIM "Paid in X seconds" → debit
    if re.search(r'paid in\s+[\d.]+\s+seconds?', text):
        return 'debit', 'paid_in_seconds'

    # 4. BHIM header "Paid" in top 20% (standalone word) → debit
    #    Real credit screenshots show "Received" in header, not "Paid"
    if image_height:
        for ln in lines:
            if ln['box'][1] < image_height * 0.20:
                t_low = ln['text'].strip().lower()
                if re.search(r'\bpaid\b', t_low) and len(t_low) <= 8 and 'received' not in t_low:
                    return 'debit', 'paid_header'

    # 5. FamPay "{Name} Paid" EOL → debit
    for ln in lines:
        m = PAID_LINE_RE.match(ln['text'].strip())
        if m:
            prefix = m.group(1).strip()
            if len(prefix) >= 3 and re.search(r'[A-Za-z]{3,}', prefix):
                return 'debit', 'paid'

    # 6. Other debit phrases
    for kw in ['sent to', 'send to', 'money sent']:
        if kw in text:
            return 'debit', kw

    # 7. Credit primary
    for kw in ['received from', 'money received', 'payment received']:
        if kw in text:
            return 'credit', kw

    # 8. Weak section headers
    if 'debited from' in text:
        return 'debit', 'debited from'
    if 'credited to' in text:
        return 'credit', 'credited to'

    # 9. BHIM To UPI ID before From UPI ID
    for ln in lines:
        if re.match(r'^to\s+upi\s+id', ln['text'].strip().lower()):
            return 'debit', 'to_upi_id'
    for ln in lines:
        if re.match(r'^from\s+upi\s+id', ln['text'].strip().lower()):
            return 'credit', 'from_upi_id'

    # 10. Standalone To/From
    for ln in lines:
        t = ln['text'].strip().lower()
        if re.match(r'^to[:\s]', t) or t.rstrip(':').strip() == 'to':
            return 'debit', 'to'
        if re.match(r'^from[:\s]', t) or t.rstrip(':').strip() == 'from':
            return 'credit', 'from'

    return 'unknown', None

# ---------------------------------------------------------------------------
# Counterparty
# ---------------------------------------------------------------------------
UI_EXCLUSIONS = [
    'hide details', 'show details', 'view details',
    'transaction details', 'payment details', 'transfer details',
    'banking name', 'message', 'utr', 'powered by',
    'unified payments', 'upi', 'click here', 'tap to',
    'share', 'download', 'more',
    'screenshot', 'screen shot',
    'split this', 'split', 'copy', 'repeat', 'pay again',
    'successful', 'success', 'failed', 'pending',
    'paid to', 'received from', 'debited from', 'credited to',
    'self transfer', 'transferred', 'money transferred',
    'paid with fam', 'paid in',
]

def _is_name(t, confidence):
    if not (2 <= len(t) <= 50):
        return False
    if confidence < 0.50:
        return False
    if not re.search(r'[A-Za-z]{3,}', t):
        return False
    if re.fullmatch(r'₹?\d[\d,]*(?:\.\d{1,2})?', t):
        return False
    low = t.lower()
    if any(kw in low for kw in UI_EXCLUSIONS):
        return False
    return True

def _name_below(lines, base_y, min_offset=20):
    below = sorted(
        [l for l in lines if l['box'][1] > base_y + min_offset],
        key=lambda x: x['box'][1],
    )
    for l in below:
        if _is_name(l['text'].strip(), l['confidence']):
            return l['text'].strip()
    return None

def _name_after_label(lines, label):
    for ln in lines:
        if label not in ln['text'].lower():
            continue
        after = re.search(rf'{re.escape(label)}\s*[:\-]?\s*(.+)$',
                          ln['text'], re.IGNORECASE)
        if after:
            candidate = after.group(1).strip()
            if _is_name(candidate, ln['confidence']):
                return candidate
        name = _name_below(lines, ln['box'][1])
        if name:
            return name
        break
    return None

def find_counterparty(lines, direction_label, direction):
    # Debit-side anchors
    if direction in ('debit', 'self_transfer'):
        name = _name_after_label(lines, 'paid to')
        if name:
            return name
        for ln in lines:
            m = PAID_LINE_RE.match(ln['text'].strip())
            if m:
                prefix = m.group(1).strip()
                if len(prefix) >= 3:
                    below = _name_below(lines, ln['box'][1])
                    if below:
                        return below
                    break
        for ln in lines:
            if re.match(r'^to\s+upi\s+id', ln['text'].strip().lower()):
                below = _name_below(lines, ln['box'][1])
                if below:
                    return below
                break

    # Credit-side anchors
    if direction == 'credit':
        name = _name_after_label(lines, 'received from')
        if name:
            return name
        for ln in lines:
            if re.match(r'^from\s+upi\s+id', ln['text'].strip().lower()):
                below = _name_below(lines, ln['box'][1])
                if below:
                    return below
                break

    # Fallback: standalone To:/From:
    for ln in lines:
        m = re.match(r'^(?:to|from)[:\s]+(.+)$', ln['text'].strip(), re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if _is_name(name, ln['confidence']):
                return name

    # Last resort: banking name
    name = _name_after_label(lines, 'banking name')
    if name:
        return name
    return None

# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def detect_status(lines, direction):
    text = ' '.join(ln['text'].lower() for ln in lines)
    for kw in ['failed', 'failure', 'declined', 'unsuccessful', 'cancelled', 'canceled']:
        if kw in text:
            return 'failed'
    for kw in ['pending', 'processing', 'in progress']:
        if kw in text:
            return 'pending'
    for kw in ['successful', 'success', 'completed', 'payment done']:
        if kw in text:
            return 'success'
    if 'paid with fam' in text:
        return 'success'
    if direction in ('debit', 'credit', 'self_transfer'):
        return 'success'
    return 'unknown'

# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------
def extract_fields(lines, img_path, image_height=None):
    amount_raw, amount_fixed, amount_conf, amount_ok = extract_amount(lines, image_height)
    date_iso, time_24h = find_date_time(lines)
    direction, dir_label = find_direction(lines, image_height)   # ← pass height
    counterparty = find_counterparty(lines, dir_label, direction)
    return {
        'image': img_path,
        'amount': amount_fixed if amount_ok else None,
        'amount_raw': amount_raw,
        'amount_conf': amount_conf,
        'accepted': amount_ok,
        'date': date_iso, 'time': time_24h,
        'direction': direction, 'direction_label': dir_label,
        'counterparty': counterparty,
        'app': detect_app(lines, img_path),
        'status': detect_status(lines, direction),
    }

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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if len(sys.argv) > 1:
    targets = []
    for arg in sys.argv[1:]:
        targets.extend(sorted(glob.glob(arg)))
else:
    targets = sorted(glob.glob('screenshots/*'))

targets = [p for p in targets if p.lower().endswith(('.jpg', '.jpeg', '.png')) and 'test_dummy' not in p]

rows = []
stats = {'color': 0, 'bw_fallback': 0, 'both_fail': 0}
total = len(targets)

for img_path in targets:
    img = Image.open(img_path)
    h = img.height

    c_lines = ocr_lines(img_path)
    if DEBUG:
        print(f"\n--- RAW OCR: {img_path} ---")
        for ln in c_lines:
            print(f"  {ln['text']!r:<50} conf={ln['confidence']:.2f} box={ln['box']}")

    c_fields = extract_fields(c_lines, img_path, image_height=h)

    if c_fields['accepted']:
        fields = c_fields
        fields['via'] = 'colour'
        stats['color'] += 1
    else:
        bw_path = os.path.join(BW_DIR, os.path.basename(img_path).rsplit('.', 1)[0] + '_bw.png')
        img.convert('L').save(bw_path)
        b_lines = ocr_lines(bw_path)
        b_fields = extract_fields(b_lines, img_path, image_height=h)
        if b_fields['accepted']:
            fields = b_fields
            fields['via'] = 'B&W fallback'
            stats['bw_fallback'] += 1
        else:
            fields = c_fields
            fields['via'] = 'manual entry'
            stats['both_fail'] += 1

    rows.append(fields)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
print(f"\n{'=' * 132}")
print(f"{'image':<24} {'amount':<10} {'direction':<14} {'to/from':<24} {'date':<12} {'time':<7} {'app':<10} {'status':<8}")
print('-' * 132)
for r in rows:
    print(f"{os.path.basename(r['image']):<24} "
          f"{str(r['amount'] or 'FAILED'):<10} "
          f"{r['direction']:<14} "
          f"{str(r['counterparty'] or '—')[:22]:<24} "
          f"{str(r['date'] or '—'):<12} "
          f"{str(r['time'] or '—'):<7} "
          f"{r['app']:<10} "
          f"{r['status']:<8}")

print(f"\n{'=' * 132}")
print("AMOUNT SUMMARY")
print('=' * 132)
print(f"Colour succeeded:       {stats['color']}")
print(f"B&W fallback rescued:   {stats['bw_fallback']}")
print(f"Both failed:            {stats['both_fail']}")
print(f"Amount success:         {stats['color'] + stats['bw_fallback']} / {total}")

print(f"\nFIELD COMPLETENESS ({total} images)")
print('=' * 132)
for field in ['amount', 'date', 'time', 'direction', 'counterparty', 'app', 'status']:
    found = sum(1 for r in rows if r[field] and r[field] not in ('unknown', None))
    pct = 100 * found / total if total else 0
    print(f"  {field:<14} {found:>3} / {total}  ({pct:.0f}%)")

failures = [r for r in rows if not r['accepted'] or r['direction'] == 'unknown' or r['status'] == 'unknown']
if failures:
    print(f"\nNEEDS ATTENTION ({len(failures)} images)")
    print('=' * 132)
    for r in failures:
        issues = []
        if not r['accepted']:
            issues.append('amount failed')
        if r['direction'] == 'unknown':
            issues.append('direction unknown')
        if r['status'] == 'unknown':
            issues.append('status unknown')
        if r['date'] is None:
            issues.append('no date')
        if r['counterparty'] is None:
            issues.append('no counterparty')
        print(f"  {os.path.basename(r['image']):<24} {', '.join(issues)}")

with open('baseline_hybrid_output.json', 'w', encoding='utf-8') as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(f"\nSaved baseline_hybrid_output.json ({total} images)")