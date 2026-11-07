## Glossary
- **PIR** — Paddle's newer intermediate representation (replaces Program IR in 3.x)
- **oneDNN** — Intel's deep-learning acceleration library; enabled by default on Linux CPU builds
- **PP-OCRv6** — current PaddleOCR model family (det = detection, rec = recognition)
- **mkldnn / oneDNN** — same thing; `enable_mkldnn` is Paddle's flag
- **`.pdparams`** — Paddle training checkpoint; `.pdmodel`/`.pdiparams` = inference graph; `.onnx` = cross-runtime export

## Pipeline concepts

- **Preference tier (amount)** — The pipeline ranks amount candidates into 4 tiers, strongest first: (1) `₹` + comma, (2) `₹` only, (3) comma only, (4) plain digits. Prevents a date or txn-ID fragment from winning over the actual amount.
- **Confidence floor** — Minimum OCR confidence (0.40) required for a numeric line to be considered as an amount candidate. Filters out icon misreads like a success checkmark OCR'd as `4` at conf 0.39.
- **Header-zone skip** — Ignores amount candidates whose bounding box sits in the top 8% of the image. Prevents status-bar digits and screen-title numbers from competing with the real amount.
- **Confidence-gated ₹ fix** — When a pure-digit amount candidate has confidence ≥ 0.90, prepend `₹`. Empirically validated: every rupee-dropped case has conf ≥ 0.94; every correctly-read-₹ case has conf ≤ 0.81, zero overlap across 13 samples.
- **B&W fallback** — If colour OCR produces no accepted amount, re-run the pipeline on a greyscale conversion. Image-specific, not systematic — fixes some images, breaks none in the current set.
- **Direction label** — The `direction_label` field records which text triggered the direction (e.g. `paid to`, `to_upi_id`, `paid_header`, `name paid`). Useful for debugging why a specific screenshot was classified debit vs credit.
- **UI_EXCLUSIONS** — A list of strings that should never be returned as counterparty (`screenshot`, `split this`, `share`, `banking name`, etc.). Prevents UI buttons from being mistaken for names.

## App-specific layout notes

- **BHIM header/footer conflict** — BHIM success screens show `" Paid"` in the header (debit indicator) but `"Payment received by BIKASH MURMU"` in the footer. Header wins — footer is descriptive narrative, header is the actual direction.
- **BHIM `To UPI ID` vs `From UPI ID`** — Both labels appear on every BHIM screen. `To UPI ID` (debit side) is checked first; `From UPI ID` (credit side) only matters when the header doesn't decide.
- **BHIM two-line time** — BHIM splits time from am/pm across two OCR lines (`11:53` on one, `am` on the next). The pipeline checks ±100px around the time line for a standalone am/pm.
- **FamPay `{Name} Paid`** — FamPay shows `Daksh Paid` with no `to`/`from`. Per project spec, this means **debit** (outgoing). Distinguished from BHIM's partial `"e Paid"` by requiring the payer prefix to be ≥3 letters.
- **PhonePe success checkmark** — PhonePe's success screen has a large green checkmark in the top-left that OCR reads as `4` at low confidence (0.37–0.55). Filtered by the confidence floor.
- **BhimPay decimal format** — BhimPay renders amounts as `₹10.00` while most other apps use `₹10`. Both are correct — the parser stores paise (integer) while preserving the original string for display.

## Stage terminology

- **Stage 1** — Python extraction pipeline. Measure accuracy, tune rules, prove approach works. Done.
- **Stage 2** — Ship to React Native. ONNX export, Expo integration, TypeScript port, parity verification.
- **Stage 3** — Fine-tune recognition model. Skipped — rule-based extraction reached 95%+ with zero training.