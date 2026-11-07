# OCR-tuning Development Log

Project: Offline UPI/payment-screenshot OCR pipeline for a React Native expense tracker.
Goal: Fine-tune PaddleOCR to correctly read `₹` and large-font amounts from real payment screenshots.
Repo: github.com/Bikash-svg/OCR-tuning (private)


## Project stages

- **Stage 1** — Python extraction pipeline: measure accuracy across real screenshots, tune rules, prove the approach works. **Done 2026-11-07.**
- **Stage 2** — Ship to React Native: ONNX export, Expo integration, port extraction to TypeScript, verify parity. **Not started.**
- **Stage 3** — Fine-tune recognition model. **Skipped** — rule-based extraction hit 95%+ across 6 apps, no training needed.

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
| D8 | 2026-11-06 | Full-image OCR; no cropping before OCR | Crop test on gpay-2 proved cropping destroys ₹ context (full image reads `₹2` at 0.79; cropped reads `2` at 1.00) | Measured |
| D9 | 2026-11-06 | Confidence-gated ₹ prepend (threshold ≥0.90) | All rupee_dropped cases have conf ≥0.94; all correct cases have conf ≤0.81 — perfect separator across 13 samples | Measured |
| D10 | 2026-11-06 | Greyscale fallback only when colour fails | B&W test: FIXED 1, LOST 0, NO CHANGE 12 — image-specific, not systematic | Measured |
| D11 | 2026-11-06 | Comma grouping as amount-identifier signal | No other numeric field on success screens uses comma grouping; both Indian lakh (`1,50,000`) and Western (`150,000`) work | Domain reasoning |
| D12 | 2026-11-07 | No fine-tuning needed for Stage 1 | Rule-based extraction reached 95%+ across 6 apps; model reads ₹ correctly when present and rendering is complete | Measured |
| D13 | 2026-11-07 | BHIM header " Paid" wins over footer "Payment received by" | Header is the real transaction direction; footer text is descriptive narrative, sometimes misleading | Debug trace on bhim-3 |
| D14 | 2026-11-07 | FamPay `{Name} Paid` → debit | Per user specification — `paid` in FamPay layout means outgoing payment | Domain confirmation |

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

Column definitions:
- **Detection rate** = an amount candidate (any number, ₹ or not) appeared in the OCR output. `NONE` = nothing found.
- **Recognition rate** = amount was correct after applying the fix column for that row.

| Date | Model | Dataset | Detection rate | Recognition rate | Dominant error | Notes |
|---|---|---|---|---|---|---|
| 2026-11-04 | PP-OCRv6_medium (stock) | `test_dummy.png` (synthetic) | 1/1 | 1/1 | none | Clean synthetic, conf=1.00 |
| 2026-11-05 | PP-OCRv6_medium (stock) | 13 real (GPay ×7, Navi ×6) | 11/13 | 7/13 | rupee_dropped (4) + not_detected (2) | Raw baseline. Confidence separator identified: dropped ₹ = conf ≥0.94, correct = conf ≤0.81, zero overlap across 13 samples. |
| 2026-11-05 | PP-OCRv6 + confidence fix (₹ prepend when conf ≥0.90) | 13 real | 11/13 | 11/13 | not_detected (2) | Rescued 4 rupee_dropped: gpay-10, gpay-2, gpay-3, navi-2. navi-3 and navi-4 still NONE. |
| 2026-11-05 | Hybrid: colour → conf fix → B&W fallback | 13 real | 12/13 | **12/13 (92%)** | manual_entry (1: navi-3) | B&W fallback rescued navi-4 (colour `7ッ` conf 0.53 → greyscale `₹4` conf 0.66). **Stage 1 final.** |



---

## Models / Artifacts

| Date | Model | Format | Size | Location | Notes |
|---|---|---|---|---|---|
| 2026-11-04 | PP-OCRv6_medium_det | Paddle (cached) | ~10 MB | `~/.paddlex/official_models/PP-OCRv6_medium_det` | Stock detector |
| 2026-11-04 | PP-OCRv6_medium_rec | Paddle (cached) | ~10 MB | `~/.paddlex/official_models/PP-OCRv6_medium_rec` | Stock recognizer (baseline) |

---

## Next

## Next

- [x] Collect 20–30 real payment screenshots (mixed apps) — **44 collected, 6 apps**
- [x] Run baseline, score detection / recognition / error type
- [x] Check if `₹` exists in `ppocr_keys_v1.txt` dictionary — **present in both PP-OCRv5 and PP-OCRv6**
- [x] Decide Stage 2: preprocessing-only vs fine-tuning — **neither; rule-based + fallback chain**
- [x] Build field extraction (direction, counterparty, date, time, app, status)
- [x] Verify across 6 apps at 95%+ coverage
- [ ] **Stage 2.1** — Export PP-OCRv6 recognizer to ONNX
- [ ] **Stage 2.2** — Test PP-OCRv6 small for mobile speed/accuracy trade-off
- [ ] **Stage 2.3** — Bundle models + dictionary in Expo app (`@mirulkhanall/rn-paddle-ocr`)
- [ ] **Stage 2.4** — Port extraction logic to TypeScript
- [ ] **Stage 2.5** — Verify 44-screenshot parity between Python and app
- [ ] **Stage 2.6** — UX polish (wait-1s nudge, verification UI, manual-entry path)




### Stage 1 result — 2026-11-05

**Raw: 7/13 (54%) → After confidence-gated fix: 11/13 (85%)**

Fix: `fix_amount(text, conf)` prepends `₹` when a pure-digit amount has conf ≥ 0.90.

Confidence is a perfect separator:
- All rupee_dropped: conf ≥ 0.94
- All correct: conf ≤ 0.81
- Zero overlap across 13 samples.

Ruled out: dictionary (₹ present in v5+v6), threshold filtering, model choice, detection split, upscaling, cropping-before-OCR, fine-tuning.

Remaining failures (2): navi-3 (`マ3`), navi-4 (`7ッ` misread 4 as 7). Both Navi, both garbage-region.

Decision: full-image OCR + confidence-gated post-process ships. No fine-tuning needed for this failure class.

### Stage 1 final summary - 2026-11-05

Raw → final: **7/13 (54%) → 12/13 (92%)** with zero training.

Three additions, each justified by a targeted measurement:

1. **Full-image OCR (not crops)** — crop test on `gpay-2` showed cropping destroys ₹ context. Crop of the amount box read `'2'` at 1.00 on a screenshot that read `'₹2'` at 0.79 in full image.
2. **Confidence-gated ₹ prepend** — `fix_amount(text, conf)` prepends `₹` when a pure-digit amount has conf ≥ 0.90. The threshold is empirical: dropped-₹ cases all ≥ 0.94, correct cases all ≤ 0.81, no overlap.
3. **Greyscale fallback** — only runs when colour fails. `baseline_bw_test.py` on the 13-image set: B&W FIXED 1 (navi-4), LOST 0, NO CHANGE 12. Colour is image-specific, not systematic — hybrid preserves the best of both.

Ruled out by measurement (full trail in Issues and Fixes):
- Dictionary missing ₹ (present in both PP-OCRv5 and PP-OCRv6 dictionaries — verified via `grep` on `inference.yml`)
- Threshold filtering (threshold 0.2 → no change in output)
- Model choice (PP-OCRv5 and PP-OCRv6 both fail the same way on gpay-2)
- Detection split (box `[334, 536, 484, 668]` includes the ₹ on gpay-2)
- Upscaling (4× and 6× upscale → detector returns 0 results; 8× → OOM)
- Fine-tuning (model reads ₹ on full-image correctly on 5/6 GPay screenshots; failure is capture-state, not capability)

**Open gap before Stage 2:** only 2 apps tested. Need 3–5 PhonePe + 3–5 Paytm screenshots to confirm the pipeline generalizes before committing to the React Native integration.


### Stage 1 final — 2026-11-07

**Pipeline:** colour OCR → confidence-gated ₹ fix → B&W fallback → field extraction.

**Scope: 44 screenshots across 6 apps (GPay, PhonePe, Navi, BHIM, FamPay, Paytm).**

| Field | Coverage |
|---|---|
| amount | 42/44 (95%) |
| date | 44/44 (100%) |
| time | 44/44 (100%) |
| direction | 44/44 (100%) |
| counterparty | 44/44 (100%) |
| app | 44/44 (100%) |
| status | 44/44 (100%) |

**Remaining failures:** `famPay-3` (amount), `navi-1` (amount) — both manual-entry cases.

**Direction rules (priority order):**
1. `self transfer` → self_transfer
2. `paid to` / `payment to` → debit
3. `Paid in X seconds` (BHIM footer) → debit
4. `Paid` header in top 20% of image (BHIM) → debit
5. `{Name} Paid` EOL (FamPay, name ≥3 chars) → debit
6. `sent to` / `send to` / `money sent` → debit
7. `received from` / `payment received` / `money received` → credit
8. `debited from` / `credited to` (weak section headers)
9. `To UPI ID` (debit) / `From UPI ID` (credit) — BHIM fallback
10. Standalone `To:` / `From:` → debit / credit

**Counterparty rules (direction-aware):**
- debit → `paid to` / `To UPI ID` / `{Name} Paid` → name below
- credit → `received from` / `From UPI ID` → name below
- fallback → `banking name`
- `screenshot` is always excluded

**Amount extraction rules (4 tiers, strongest first):**
1. ₹ + comma (e.g. `₹7,576`)
2. ₹ only (e.g. `₹700`)
3. comma only (e.g. `7,576`)
4. plain digits (position + confidence fix)

Confidence floor 0.40; header-zone skip y ≥ 8% of image; if pure-digit amount has conf ≥ 0.90, prepend `₹`.

**Date/time patterns supported:**
- `11 Oct 2026`, `5th Nov 26`, `5 Nov 2026` (month-name)
- `Nov 04, 2026` (US order)
- `04/11/2026`, `04-11-2026` (numeric)
- `11:02 am` / `11:02am` (same-line am/pm)
- `11:53` + `am` on next line (BHIM two-line format)

**Bugs found and fixed during Stage 1:**
- Status-bar digits misread as amount → confidence floor + header-zone skip
- PhonePe success checkmark `4` misread as amount → confidence floor
- Comma-grouped amounts competing with dates/txn IDs → comma-preference tier
- BHIM contradictory header/footer signals → header wins
- BHIM `5th Nov 26` date format → ordinal + 2-digit year patterns
- FamPay `{Name} Paid` without `to`/`from` → debit; guard against BHIM's partial `e Paid`
- `screenshot` appearing as counterparty → added to UI_EXCLUSIONS