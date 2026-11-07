# OCR-tuning

Offline OCR pipeline that extracts structured transaction data from Indian UPI payment screenshots. Built for a React Native expense tracker that must work fully offline — no cloud calls, no Google Play Services dependency for the OCR layer.

**Status:** Stage 1 complete — 44 screenshots across 6 apps, all fields ≥95%, zero fine-tuning. Stage 2 (React Native integration) not started. See [LOG.md](./LOG.md) for the full decision trail.

## What it does

Input: a payment success screenshot (jpg/png)
Output: a structured transaction:

    {
      "amount": "₹7,576",
      "direction": "debit",
      "counterparty": "SUMIT DUTTA",
      "date": "2026-10-11",
      "time": "11:02",
      "app": "PhonePe",
      "status": "success"
    }

## Apps covered

- GPay
- PhonePe
- Navi
- BHIM
- FamPay
- Paytm

## Pipeline

1. **OCR** — PaddleOCR PP-OCRv6 (medium) on the full colour image
2. **Amount extraction** — 4 preference tiers (₹+comma > ₹ > comma > plain) + confidence floor 0.40 + header-zone skip
3. **Confidence-gated ₹ fix** — prepend ₹ when pure-digit amount has conf ≥ 0.90
4. **Greyscale fallback** — re-run in B&W when colour produces no usable amount
5. **Field extraction** — direction, counterparty, date, time, app, status via rule-based label anchoring

## Accuracy

| Field | Coverage (44 screenshots / 6 apps) |
|---|---|
| amount | 42/44 (95%) |
| date | 44/44 (100%) |
| time | 44/44 (100%) |
| direction | 44/44 (100%) |
| counterparty | 44/44 (100%) |
| app | 44/44 (100%) |
| status | 44/44 (100%) |

Remaining failures: `famPay-3` (amount), `navi-1` (amount) — both honest manual-entry cases.

## Requirements

- Ubuntu (or any Linux with Python 3.9–3.13; **Python 3.14 not supported by PaddlePaddle**)
- Python 3.12 (installed via deadsnakes PPA)
- venv scoped to the project folder

## Setup

    # System libs (one-time)
    sudo apt install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 python3.12 python3.12-venv python3.12-dev

    # Project venv
    cd ~/my_things/code-stuff/projects/OCR-train
    python3.12 -m venv venv
    source venv/bin/activate

    # Dependencies
    pip install --upgrade pip
    pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    pip install paddleocr

## Running

    source venv/bin/activate
    python baseline_hybrid.py                          # all screenshots/
    python baseline_hybrid.py screenshots/gpay-2.jpg   # one image
    python baseline_hybrid.py screenshots/bhim-*.jpg   # glob

Set `DEBUG_OCR=1` to dump raw OCR lines per image before the summary table.

## Files

| File | Purpose |
|---|---|
| `baseline_hybrid.py` | **Production pipeline** (OCR + amount + fields) |
| `baseline_test.py` | Diagnostic: initial baseline measurements |
| `baseline_bw_test.py` | Diagnostic: greyscale test (colour is image-specific) |
| `crop_test.py` | Diagnostic: crop experiment (proved cropping is harmful) |
| `compare_test.py` | Diagnostic: same-amount A/B capture test |
| `baseline_scores.csv` | Per-image scoring spreadsheet |
| `LOG.md` | Chronological decision log with measurements |
| `glossary.md` | PaddleOCR / pipeline terminology |

## Known gaps

- Only **success** screens tested; pending/failed layouts not covered
- `navi-1.jpg` and `famPay-3.jpg` fail amount extraction (2/44 = 4.5%) — manual entry
- Comma thousands only tested on PhonePe; BhimPay decimal format tested (`₹10.00`)
- Field extraction is rule-based, not learned — some future app will need a new rule

## Never committed

Screenshots, model weights (`.onnx`), and OCR output JSONs are gitignored. See `.gitignore`.

## Related

- `LOG.md` — why the pipeline looks the way it does (decisions, measurements, bugs fixed)
- `glossary.md` — terminology reference (PIR, oneDNN, PP-OCRv6, etc.)