# interfaze-consistency-eval

**Gemini 3.5 Flash scored higher than Interfaze on every field under strict scoring** (all 4 fields
correct: 41.6% vs 32.8% of 125 receipt-runs). Per field the margin is 2–9 receipt-runs, less than 2
receipts' worth. Both models got every price value right once formatting was ignored.

**Interfaze was more consistent byte for byte** (5 identical outputs on 84% vs 68% of receipts).
Most of Gemini's extra variation is formatting, and after normalisation the gap is 84% vs 80%
(21 vs 20 receipts). Neither model gave the same content every time at temperature 0.

This is a 25-receipt pilot on one domain: differences of one or two receipts are within noise.
Interfaze was slower (median 12.9 s vs 4.6 s) and cost about 2.3× more per call here.

## Method

- **Task:** extract `total_price`, `subtotal_price` and the item names from a photo of a receipt,
  as JSON.
- **Dataset:** the first 25 receipts of the CORD v2 test split (`naver-clova-ix/cord-v2`), which
  are Indonesian shop and restaurant receipts. The ground truth is human-annotated and verbatim:
  typos, punctuation and thousands separators are kept as printed (checked by eye on 5 receipts,
  see [DECISIONS.md](DECISIONS.md#3-ground-truth-check-is-the-ground-truth-verbatim)).
- **Models:** `interfaze-1` (Interfaze) vs `gemini-3.5-flash` (Google), one of the two Gemini
  models Interfaze's paper reports beating. Both are called through the same OpenAI-compatible
  client with an identical request body.
- **Fields scored (4):** total price (skipped for the 1 receipt that prints no total), subtotal price
  (skipped for the 10 receipts that print none), number of items, and item names (compared as a
  multiset).
- **Prompt:** the same for both models. It ends with: *"Copy every value exactly as printed on the
  receipt, including currency symbols, commas and dots. Do not reformat."* The same JSON schema is
  enforced through `response_format` (`strict: true`).
- **Scoring, twice:**
  - **STRICT:** an exact string match after trimming outer whitespace. Did the model copy what is
    printed?
  - **LENIENT:** prices reduced to digits (`"Rp 45.500"` → `45500`), names lowercased and trimmed.
    Is the value right?
- **Repeats:** `temperature = 0`, and each receipt is sent **5 times** to each model (25 × 5 × 2 =
  250 calls). Interfaze's response cache is switched off with its documented
  `x-interfaze-bypass-cache` header, so that each run is a fresh inference
  ([why](DECISIONS.md#8-interfaze-docs-check-interfazeaidocs-read-2026-11-01)).
- **Consistency** is measured on raw output (byte-identical JSON text) and on normalised output
  (after LENIENT normalisation).

## Results

25 receipts × 5 runs per model, run on 2026-11-01. Full metrics are in
[results/summary.csv](results/summary.csv), and every call is in
[results/per_run.csv](results/per_run.csv).

| | Interfaze `interfaze-1` | Gemini `gemini-3.5-flash` |
|---|---|---|
| **Accuracy, % of receipt-runs** (strict / lenient) | | |
| total_price (120 runs) | 87.5 / 100.0 | 93.3 / 100.0 |
| subtotal_price (75 runs) | 93.3 / 100.0 | 96.0 / 100.0 |
| item_count (125 runs) | 89.6 / 89.6 | 96.0 / 96.0 |
| item_names (125 runs) | 40.8 / 40.8 | 48.0 / 48.0 |
| all 4 fields correct (125 runs) | 32.8 / 40.8 | 41.6 / 48.0 |
| **Consistency, % of 25 receipts** (raw / normalised) | | |
| all 5 outputs identical | **84 / 84** | **68 / 80** |
| distinct outputs per receipt, mean (max) | 1.2 (2) / 1.2 (2) | 1.4 (3) / 1.2 (3) |
| total_price agrees in all 5 runs | 100 / 100 | 96 / 100 |
| subtotal_price agrees in all 5 runs | 100 / 100 | 96 / 100 |
| item_count agrees in all 5 runs | 92 / 92 | 100 / 100 |
| item_names agree in all 5 runs | 84 / 84 | 80 / 80 |
| **Other** | | |
| schema-valid responses | 100% | 100% |
| latency, median / p90 | 12.9 s / 76.4 s | 4.6 s / 9.3 s |
| estimated cost per call (all 125 calls) | $0.019 ($2.39) | $0.008 ($1.04) |

Notes on reading the table:

- **Consistency is agreement with itself, not with the ground truth.** Interfaze returned the same
  total on receipt 013 in all 5 runs, and it was wrong under STRICT scoring every time. That still
  counts as consistent.
- **The raw vs normalised gap is formatting.** On 2 receipts, Gemini returned the same content but
  sometimes pretty-printed the JSON and sometimes compacted it. On 1 receipt it switched between
  `51,300` and `51.300`. If whitespace in the JSON is ignored, Gemini's raw consistency would be
  76% (19/25). Interfaze's raw and normalised numbers are equal: its variation is all in content.
- **Item names are low for both models, mostly because of a labelling convention.** Of the
  item-name failures, 46/74 (Interfaze) and 45/65 (Gemini) differ from the ground truth only by a
  leading or trailing number, i.e. a quantity or item code printed on the same line (`"1 JASMINE
  MT ( L )"`, `"901016 -TICKET CP"`). CORD stores those separately from the name. The prompt said
  "copy exactly as printed", so this is a conflict between the instruction and the labels, and it
  affects both models about equally. On one receipt (016), CORD has `IVINERAL WATER` for printed
  `MINERAL WATER`, and both models read the real word.
- **Latency was measured at different times.** Interfaze's calls ran 14:17–15:55 UTC; 31 of them
  took over 30 s, all in runs 2–4. Gemini's calls all ran 16:07–16:10 UTC (it was delayed by a
  quota issue, see [DECISIONS.md](DECISIONS.md#9-gemini-model-and-endpoint)).
- **Cost:** Interfaze's per-token price is lower, but it reported about 11,000 input tokens per
  receipt vs about 1,200 for Gemini, because its OCR pre-pass counts as input. Gemini's output
  includes its hidden thinking tokens.

### Three failure examples

**1. `cord_test_002`: Interfaze reads a letter as a digit in 2 of 5 runs.**

| | total_price | item names |
|---|---|---|
| Ground truth | `28,000` | `JASMINE MT ( L )` |
| Interfaze | `28.000` ×5 | `1 JASMINE MT ( L )` ×3, `1 JASMINE MT ( 1 )` ×2 |
| Gemini | `28,000` ×5 | `JASMINE MT ( L )` ×2, `1 JASMINE MT ( L )` ×2, `1  JASMINE MT ( L )` ×1 (two spaces) |

Interfaze consistently writes the comma as a dot and, at temperature 0, reads the size `L` as `1`
in two runs. Gemini gets the total right but produces three different versions of the item name.

**2. `cord_test_016`: Interfaze's item list changes between runs.** The receipt lists a drink,
three zero-priced toppings (LONGAN, ROASTED ALMOND, MANGGO) and a water. CORD stores the toppings
as sub-items of the drink, so the ground truth has 2 items: `[LG] BLACK SAKURA`,
`IVINERAL WATER (bundling) 5k`.

| | item_count | item names |
|---|---|---|
| Interfaze | 2 ×3, **5 ×2** | `1 [LG] BLACK SAKURA`, `1 MINERAL WATER (bundling) 5k`; in 2 runs also `1 LONGAN`, `1 ROASTED ALMOND`, `1 MANGGO` |
| Gemini | 2 ×5 | `1 [LG] BLACK SAKURA`, `1 MINERAL WATER (bundling) 5k` ×5 |

The prompt says not to list toppings as separate items; Interfaze follows that in 3 of 5 runs.
Neither model matches the names: both add the quantity `1`, and both read `MINERAL` where the
ground truth has `IVINERAL`.

**3. `cord_test_013`: consistent and wrong vs. inconsistent and sometimes right.** This is a
dot-matrix receipt where the thousands separator is hard to see.

| | total_price | subtotal_price |
|---|---|---|
| Ground truth | `51.300` | `46.636` |
| Interfaze | `51,300` ×5 | `46,636` ×5 |
| Gemini | `51,300` ×3, `51.300` ×2 | `46,636` ×3, `46.636` ×2 |

Interfaze is perfectly consistent and wrong under STRICT scoring in all 5 runs. Gemini is
inconsistent but matches the ground truth in 2 runs. Under LENIENT scoring, both are 100% correct
and 100% consistent.

## Reproduce

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env              # add INTERFAZE_API_KEY and GEMINI_API_KEY
.venv/bin/python -m pytest        # metric tests
.venv/bin/python run_eval.py --n 25 --runs 5
```

`run_eval.py` downloads and caches the receipts under `data/` (gitignored). It saves every raw API
response to `results/raw/{model}/{receipt_id}_run{k}.json` and skips calls that are already saved,
so an interrupted run resumes where it stopped. It then writes `results/summary.csv` (all metrics)
and `results/per_run.csv` (one row per call with extracted values, ground truth and correctness).
To re-score the committed raw responses without calling any API:
`run_eval.py --score-only`.

| File | What it does |
|---|---|
| `data.py` | loads and caches the first N CORD test receipts |
| `extract.py` | the prompt, schema, model configs, and the API call with retries |
| `metrics.py` | pure scoring functions (normalisation, accuracy, consistency, latency, cost) |
| `run_eval.py` | collects responses (resume-safe), scores them, writes the CSVs |
| `tests/` | pytest tests for every metric function |
| `DECISIONS.md` | every design choice and why |

## Limitations

- **Small sample:** 25 receipts. One receipt is 4 percentage points of a receipt-level metric, and
  most accuracy differences in the table amount to 1–2 receipts. Treat them as indications, not
  rankings.
- **One domain:** printed Indonesian receipts in rupiah. The LENIENT price rule (keep digits)
  relies on rupiah having no decimals. Results may not carry over to invoices, forms, handwriting
  or other languages.
- **One setting:** `temperature = 0` only. Google recommends temperature 1.0 for Gemini 3 models
  and warns that lower values may degrade output, so this setting may not be Gemini's best.
  Defaults were kept for everything else: Gemini thinks by default, Interfaze runs an OCR pre-pass
  by default.
- **A single day's run** (2026-11-01). Both services can change models or infrastructure without
  notice. Consistency across days is not measured.
- **The models ran at different times.** All 125 Gemini calls ran within 3 minutes, while
  Interfaze's 125 calls were spread over 1 h 40 min. This affects the latency comparison, and
  repeats close together in time could be more alike.
- **Which Gemini model served the calls is not fully certain.** We requested `gemini-3.5-flash`
  and the API echoed it, but a quota error charged a request to `gemini-3.6-flash`
  ([details](DECISIONS.md#9-gemini-model-and-endpoint)).
- **Interfaze's cache was bypassed.** A default-settings user sending an identical request again
  would often get a cached copy (`vcache: true`), which is perfectly consistent by construction.
  We measure the model, not the cache.
- **Ground-truth conventions.** CORD joins words with single spaces, stores quantities and item
  codes apart from names, and has at least one literal misreading (`IVINERAL`). A model can be
  faithful to the paper and still fail STRICT. Subtotals that a model invents for receipts without
  one are not penalised.
- **Cost is an estimate** from reported token counts and list prices (Gemini 3.5 Flash's price is
  from a web search, because Google's pricing page now lists its successor).
