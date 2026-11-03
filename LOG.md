# OCR-tuning Development Log

Project: Offline UPI/payment-screenshot OCR pipeline for a React Native expense tracker.
Goal: Fine-tune PaddleOCR to correctly read `₹` and large-font amounts from real payment screenshots.
Repo: github.com/Bikash-svg/OCR-tuning (private)

---

## Decisions

| ID | Date | Decision | Rationale | Decided by |
|---|---|---|---|---|
| D1 | 2026-11-04 | Use PaddleOCR (Option A) over ML Kit + custom recognizer (Option B) | ML Kit fails at *detection* of large-font amounts, not just recognition; Option B cannot fix detection | Bikash + assistant |
| D2 | 2026-11-04 | Keep system Python 3.14 as default; use 3.12.16 via venv | PaddlePaddle 3.3.0 has no 3.14 wheels; venv scopes 3.12 to this folder only | Bikash |
| D3 | 2026-11-04 | Install 3.12.16 from deadsnakes PPA | Needed `python3.12-venv` and `-dev`; resolves apt naming | Bikash |
| D4 | 2026-11-04 | Pin `paddlepaddle==3.3.0` + `paddleocr==3.7.0` | Latest stable with 3.12 support | Bikash |
| D5 | 2026-11-04 | Set `enable_mkldnn=False` in `PaddleOCR()` | Works around Paddle 3.3.0 oneDNN PIR regression (Issue #77340) | Bikash |
| D6 | 2026-11-04 | Fine-tuning in Python on laptop; app only receives final `.onnx` | Separates training (heavy) from inference (light); keeps app 100% React Native | Bikash |
| D7 | 2026-11-04 | Make `OCR-tuning` repo private | Screenshots contain personal financial data | Bikash |

---

## Log

| Date | Track and step | What changed | Files |
|---|---|---|---|
| 2026-11-04 | Setup — Python env | Installed Python 3.12.16 via deadsnakes; resolved dpkg conflict between Ubuntu `python3.12-minimal` (3.12.3) and PPA `python3.12` (3.12.16) | (system packages) |
| 2026-11-04 | Setup — venv | Created `venv/` with Python 3.12 in project dir | `venv/` (gitignored) |
| 2026-11-04 | Setup — deps | Installed `paddlepaddle==3.3.0`, `paddleocr==3.7.0`, OpenCV 4.10.0 | `venv/` |
| 2026-11-04 | Setup — git | `git init -b main`, remote → `Bikash-svg/OCR-tuning`, `.gitignore` protects screenshots/venv/models | `.gitignore` |
| 2026-11-04 | Stage 1 — harness | Created `baseline_test.py` — OCRs all images in `screenshots/`, dumps to `baseline_output.json` | `baseline_test.py` |
| 2026-11-04 | Stage 1 — API fix | PaddleOCR 3.x: `use_angle_cls` → `use_textline_orientation`; `.ocr(cls=True)` → `.predict()` | `baseline_test.py` |
| 2026-11-04 | Stage 1 — oneDNN fix | Added `enable_mkldnn=False` to bypass Paddle 3.3.0 CPU regression | `baseline_test.py` |
| 2026-11-04 | Stage 1 — smoke test | Synthetic image → `'Paid Rs 280.00 to Swiggy'` (conf=1.00); pipeline confirmed working | `screenshots/test_dummy.png` (gitignored) |

---

## Issues and Fixes

| Date | Problem | Root cause | Fix | Refs |
|---|---|---|---|---|
| 2026-11-04 | `dpkg` "trying to overwrite /usr/bin/python3.12" | Ubuntu `python3.12-minimal` (3.12.3) conflicts with PPA `python3.12` (3.12.16) | `dpkg --force-depends -r libpython3.12-minimal python3.12-minimal` + `apt --fix-broken install` | — |
| 2026-11-04 | `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]` | Paddle 3.3.0 oneDNN PIR conversion bug (CPU inference) | `PaddleOCR(..., enable_mkldnn=False)` | PaddlePaddle Issue #77340 |
| 2026-11-04 | `TypeError` on `use_angle_cls` / `cls=True` | PaddleOCR 3.x renamed arguments | `use_textline_orientation=True`; use `.predict(img)` | PaddleOCR 3.x migration |
| 2026-11-04 | `libGL.so.1: cannot open shared object file` (potential) | Ubuntu minimal lacks OpenCV system libs | `sudo apt install libgl1 libglib2.0-0 libsm6 libxext6 libxrender1` | — |

---

## Measurements / Results

| Date | Model | Dataset | Detection rate | Recognition rate | Dominant error | Notes |
|---|---|---|---|---|---|---|
| 2026-11-04 | PP-OCRv6_medium (stock) | `test_dummy.png` (synthetic) | 1/1 | 1/1 | none | Clean synthetic, conf=1.00 |
| 2026-11-04 | PP-OCRv6_medium (stock) | 20–30 real screenshots | TBD | TBD | TBD | Pending collection |

---

## Models / Artifacts

| Date | Model | Format | Size | Location | Notes |
|---|---|---|---|---|---|
| 2026-11-04 | PP-OCRv6_medium_det | Paddle (cached) | ~10 MB | `~/.paddlex/official_models/PP-OCRv6_medium_det` | Stock detector |
| 2026-11-04 | PP-OCRv6_medium_rec | Paddle (cached) | ~10 MB | `~/.paddlex/official_models/PP-OCRv6_medium_rec` | Stock recognizer (baseline) |

---

## Next

- [ ] Collect 20–30 real payment screenshots (mixed apps, large font, ₹ visible, commas)
- [ ] Run baseline, score detection / recognition / error type per screenshot
- [ ] Check if `₹` exists in `ppocr_keys_v1.txt` dictionary
- [ ] Decide Stage 2: preprocessing-only vs fine-tuning (or both)
- [ ] Build cropped amount-line dataset (`image_path\tlabel` format)
- [ ] Add `₹` to character dictionary if missing
- [ ] Fine-tune PP-OCRv6_medium_rec
- [ ] Export to ONNX, verify mobile integration
