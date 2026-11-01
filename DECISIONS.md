# Design decisions

Every choice in this benchmark, with the reason for it. The scope is version 1: one task, one
dataset, two models, a time box of about 90 minutes. Anything left out is listed under
"Not done (on purpose)" at the end.

## 1. What is being measured, and why

Interfaze markets `interfaze-1` for "deterministic" tasks (OCR, structured extraction) and claims
the "highest accuracy, precision and consistency". Its paper (arXiv 2602.04101) reports accuracy
(OCRBench v2 70.7%, olmOCR 85.7%) against Gemini 3 / 3.5 Flash, but no run-to-run consistency
figures. This repo measures both on one concrete task: **pull the total, subtotal and item names
out of a receipt photo**. If you send the same image with the same request five times, do you get
the same, correct answer five times?

## 2. Dataset: CORD v2, first 25 test receipts

- **Why CORD.** It is a public, widely used receipt dataset (`naver-clova-ix/cord-v2` on Hugging
  Face). Its ground truth is annotated per word box, so it records text as printed. Receipt
  extraction is exactly the kind of "deterministic" task Interfaze targets.
- **Why the first 25 of the test split.** Taking them in dataset order makes the sample fixed and
  reproducible. Nobody picked receipts that favour either model. 25 receipts × 5 runs × 2 models is
  250 calls, which fits the time box.
- **Images are sent as the original JPEG bytes.** `data.py` loads with `Image(decode=False)`, so
  nothing is re-encoded or resized. The largest image (`cord_test_005`) is 7.7 MB, about 10 MB as
  base64. That is under both APIs' 20 MB inline limit.
- **Cache.** Images and `gt_parse` are saved under `data/cord_test/`. That folder is gitignored and
  can be re-downloaded.

## 3. Ground-truth check (is the ground truth verbatim?)

Before writing any scoring code I compared `gt_parse` with the image for receipts 000–004 by eye.
I also scanned the structure of all 25.

**The values are verbatim as printed.** The ground truth keeps:
- typos: `ICE BLACKCOFFE`, `CHIKEN KATSU FF` (receipt 004)
- punctuation: `-TICKET CP` (000), `J.STB PROMO`, `Y.B.BAT` (001)
- each receipt's own thousands separator: `60.000` (000), `24,000` (002), `17500` (001)
- the currency prefix when it is printed next to the total: `Rp. 91,000` (017), `Rp 38.000` (021),
  `Rp.118.000` (024)

This is why STRICT scoring (exact text) is meaningful on this dataset at all.

Caveats found:
- **Whitespace.** CORD joins word boxes with single spaces. A wide gap on paper therefore becomes
  one space: receipt 002 prints `JASMINE MT  ( L )`, and the ground truth has `JASMINE MT ( L )`. A
  model that reproduces the paper spacing exactly can fail STRICT.
- **The menu is a dict when there is one item and a list otherwise** (10 of 25 receipts).
  `metrics.gt_fields` wraps the dict in a list.
- **Receipt 009 has no TOTAL line.** It prints Sub Total, Tunai (cash) and Kembalian (change), so
  the ground truth has no `total_price`. Just as the spec says for subtotal, a field missing from
  the ground truth is skipped for that receipt.
- **Receipt 013's subtotal is annotated twice** (`["46.636", "46.636"]`) because the receipt prints
  it twice. We take the first value. Both are identical.
- **Sub-items.** 4 of 25 receipts have items nested under another item (e.g. receipt 002:
  `COCONUT JELLY ( L )` under `JASMINE MT ( L )`). CORD counts them as part of the parent, not as
  separate menu entries. See the prompt decision below.
- **Prices have no decimal part.** All 39 ground-truth prices (24 totals, 15 subtotals) are whole rupiah, and `.` and `,`
  are only ever thousands separators. This is what makes the "digits only" LENIENT rule valid
  (section 6).

## 4. The four fields

| Field | Ground truth | Note |
|---|---|---|
| `total_price` | `gt_parse.total.total_price` | skipped where missing (1 receipt) |
| `subtotal_price` | `gt_parse.sub_total.subtotal_price` | skipped where missing (10 receipts) |
| `item_count` | number of entries in `gt_parse.menu` | top-level entries only |
| `item_names` | list of `menu[].nm` | compared as a multiset (order ignored, duplicates count) |

They cover single-value OCR (two prices), structure (how many items) and list OCR (names). Item
prices, quantities and dates are left out to keep v1 small.

If a model returns a subtotal for a receipt whose ground truth has none, that is not scored, so a
hallucinated subtotal goes unpenalised. This is a limitation that follows from the skip rule.

## 5. The request: prompt, schema, parameters

**Both models get the same prompt and the same JSON schema in the same request body**
(`extract.py`). The only differences are model ID, base URL, API key and one HTTP header
(section 8).

Prompt (`extract.PROMPT`):

```
You are extracting data from a photo of a store receipt.

Return JSON with these fields:
- total_price: the final total amount printed on the receipt.
- subtotal_price: the subtotal amount printed on the receipt, or null if no subtotal is printed.
- items: one entry per purchased line item, in the order printed, with its name. Do not list sub-items, toppings or modifiers printed beneath an item as separate entries.

Copy every value exactly as printed on the receipt, including currency symbols, commas and dots. Do not reformat.
```

- The last paragraph is the required verbatim instruction, word for word.
- **"or null if no subtotal is printed"** matches the schema (`string | null`). It gives the model a
  legitimate answer when there is no subtotal.
- **"Do not list sub-items ... as separate entries"** was added after the ground-truth check,
  before any model output was seen. It aligns the task with how CORD defines a menu entry, so
  `item_count` measures extraction, not a disagreement over definitions.
- **The prompt was frozen before any scoring and was not tuned on model outputs.** The smoke test
  showed both models writing receipt 000's item as `901016 -TICKET CP`. On the receipt, 901016 is
  the item code, which CORD stores separately in `num`; the ground-truth name is `-TICKET CP`. I
  did not add a rule against this. Changing the prompt after seeing test outputs would be fitting
  to the test set.

Schema (`extract.SCHEMA`), sent as OpenAI-style structured output:

```json
{"type": "json_schema",
 "json_schema": {"name": "receipt", "strict": true,
   "schema": {"type": "object",
     "properties": {
       "total_price":    {"type": "string"},
       "subtotal_price": {"type": ["string", "null"]},
       "items": {"type": "array", "items": {"type": "object",
                 "properties": {"name": {"type": "string"}},
                 "required": ["name"], "additionalProperties": false}}},
     "required": ["total_price", "subtotal_price", "items"],
     "additionalProperties": false}}}
```

- `strict: true` and `additionalProperties: false` ask both APIs for the tightest schema
  enforcement they offer. Interfaze's default is `strict: false`.
- Both APIs accepted the nullable type array `["string", "null"]` (checked in the smoke test).

Request parameters (every call; the full request is saved inside each raw file, with the image
replaced by its byte size and SHA-256):

| Parameter | Value | Why |
|---|---|---|
| `temperature` | `0` | the most favourable setting for determinism (see caveat below) |
| `response_format` | the schema above | same for both |
| `messages` | one user message: prompt text + `image_url` with a base64 data URL | same for both |
| `seed` | not sent | Interfaze's docs mention it in the `temperature` description, but it isn't in the documented parameter list |
| `max_tokens`, `top_p` | not sent (API defaults) | not needed; fewer knobs to explain |
| `reasoning_effort` | not sent (API defaults) | Interfaze: reasoning off by default. Gemini: thinking on by default and cannot be switched off for Gemini 3 models |

**Caveat on temperature 0 for Gemini.** Google's Gemini 3 developer guide says: "For all Gemini 3
models, we strongly recommend keeping the temperature parameter at its default value of `1.0`" and
"Changing the temperature (setting it below 1.0) may lead to unexpected behavior, such as looping
or degraded performance". The guide covers Gemini 3 Flash and does not mention 3.5. We used 0
anyway, as the experiment specifies: it is the setting a user who wants consistency would reach
for. The Gemini results should still be read with Google's advice in mind.

**Caveat on thinking.** Gemini thinks before answering (a hidden, billed reasoning step); Interfaze,
by default, does not reason but runs an OCR pre-pass (its `precontext`). Both are each vendor's
default behaviour, and changing either was out of scope. It affects latency and cost.

## 6. STRICT vs LENIENT scoring

Every field is scored twice (`metrics.py`):

- **STRICT:** exact string match after trimming outer whitespace only. This measures OCR fidelity:
  did the model copy what is printed? The prompt explicitly asks for that.
- **LENIENT:** prices keep digits only (`"45,500"` → `45500`, `"Rp 45.500"` → `45500`), and names
  are lowercased and trimmed. This measures value correctness: is the number right even if the
  formatting differs?

Why digits only is valid here: every CORD price is in whole rupiah (checked on all 39 ground-truth prices), so
dots and commas are thousands separators. **The same rule would be wrong for currencies with
cents** (e.g. `12.50` → 1250).

`item_count` is an integer, so STRICT and LENIENT are the same for it.

No other normalisation is applied (for example, internal whitespace is not collapsed). The spec
defines LENIENT as exactly the rules above, and adding rules after seeing outputs would make the
metric drift.

## 7. Metric definitions

- **Field accuracy:** for each field, the % of (receipt, run) pairs where the field is correct.
  Pairs whose ground truth lacks the field are skipped. A response that is not a JSON object counts
  as wrong on every field.
- **All-fields accuracy:** the % of (receipt, run) pairs where every field that has ground truth is
  correct.
- **Raw consistency:** compares `choices[0].message.content`, i.e. the JSON text the model
  produced, byte for byte. Not the whole HTTP response, which contains ids and timestamps that
  always differ.
- **Normalised consistency:** compares the parsed output after LENIENT normalisation (prices as
  integers, names lowercased and sorted). The gap between raw and normalised consistency shows how
  much of the variation is only formatting.
- **(a) Exact-output consistency:** the % of receipts where all 5 outputs are identical.
- **(b) Per-field agreement:** the % of receipts where one field is identical across all 5 runs.
  Raw uses the value exactly as returned (item order kept); normalised uses the LENIENT value. This
  is agreement with itself, not with the ground truth: a model can agree 5/5 on a wrong answer.
- **(c) Distinct outputs per receipt:** how many different outputs the 5 runs produced (1 is
  perfectly consistent, 5 means every run differed). Reported as mean and max.
- Only receipts with all 5 runs saved count towards consistency, so every receipt is judged on the
  same number of tries. Accuracy uses every saved run.
- **Schema validity:** the response parses as JSON and validates against the schema with
  `jsonschema`. A response wrapped in ```` ```json ```` fences counts as invalid, because that is
  not what a client asking for structured output can parse directly.
- **Latency:** wall-clock time of the successful HTTP call (median and p90). Time spent waiting
  between retries is excluded. Calls ran 4 at a time, so latency includes each API's own queueing
  under light load.
- **Cost:** `prompt_tokens × input price + (total_tokens − prompt_tokens) × output price`. Output
  is computed from the total rather than `completion_tokens` because Gemini's OpenAI-compatible
  endpoint leaves its billed thinking tokens out of `completion_tokens` (smoke test: 40 completion
  tokens, but 2,350 total vs 1,202 prompt). Prices, USD per 1M tokens:
  - Interfaze `interfaze-1`: $1.50 in / $3.50 out (interfaze.ai/pricing). Interfaze bills images as
    input tokens, and its reported prompt tokens include its OCR pre-pass (~11k for one receipt).
  - Gemini 3.5 Flash: $1.50 in / $9.00 out (per a web search on 2026-11-01; Google's pricing page
    no longer lists 3.5 Flash because it has been superseded by 3.6 Flash). Treat it as an estimate.

## 8. Interfaze docs check (interfaze.ai/docs, read 2026-11-01)

- **Image input:** OpenAI-style `{"type": "image_url", "image_url": {"url": ...}}`. The URL can be
  "Publicly accessible URL or base64 data URL (`data:image/jpeg;base64,...`)". We use base64 for
  both models, so neither has to fetch a URL.
- **Structured output:** `response_format` "Follows the OpenAI structured output specification":
  `{"type": "json_schema", "json_schema": {"name", "schema", "strict"}}`, with `strict` defaulting
  to false. `message.content` comes back as a JSON string. No differences from OpenAI were needed.
- **Non-standard response fields:** `precontext` (the output of Interfaze's OCR pre-pass, with
  text, sections and word boxes) and `vcache` ("Whether the response was served from the model's
  verified cache"). Raw files keep both.
- **The cache decision.** In the smoke test, an identical second request came back with
  `vcache: true`: byte-identical output in 5.7 s instead of 13.1 s, with fewer tokens billed. With
  default settings, runs 2–5 would mostly be cache replays, and the consistency score would measure
  the cache rather than the model. After discussing it, we send the documented header
  `x-interfaze-bypass-cache: true` on every Interfaze call. We verified that it works (two repeat
  calls returned `vcache: false` with full token counts), and `summary.csv` reports `vcache_hits`,
  which should be 0. Gemini has no response cache (its implicit caching only discounts repeated
  prompt tokens), so this puts the two models on the same footing. **Note for readers:** a
  customer using default settings would see the cache's consistency on repeated identical
  requests, which is higher than what we measure.
- **Rate limit:** 50 requests/s, far above what we use.

## 9. Gemini model and endpoint

- **Model: `gemini-3.5-flash`** (exact ID). It is one of the models Interfaze's paper claims to
  beat, and it was listed on the key (`models.list()`), so no fallback to Gemini 3 Flash was
  needed. The key also lists newer Flash models (3.6, 3.7, 3.8); we deliberately stayed with the
  paper's comparison point.
- **Endpoint: Gemini's OpenAI-compatible endpoint**
  (`https://generativelanguage.googleapis.com/v1beta/openai/`) rather than the `google-genai` SDK.
  Both models then go through the same `openai` client code with a byte-for-byte identical request
  body, which is the simplest way to show the comparison is like-for-like. Trade-off: the
  compatibility layer is a translation, so Gemini-only options (e.g. `thinking_level`) aren't
  directly exposed. We use none of them.
- **Which model actually served the calls.** Every response echoes `"model": "gemini-3.5-flash"`,
  and `models.retrieve("gemini-3.5-flash")` returns "Gemini 3.5 Flash". However, when the original
  free-tier key hit its quota, the 429 error charged the request to `model: gemini-3.6-flash`. So
  Google may now serve 3.5 Flash requests on 3.6 Flash (its successor), or the two may just share a
  quota bucket. We cannot tell which from outside. We report what we requested and what the API
  echoed, and flag this uncertainty.
- **Key and billing tier.** The first Gemini key was on the free tier (20 requests/day per model),
  which ran out after 9 benchmark calls. All 125 reported Gemini calls were then made with a second
  key on a prepaid project. The 9 free-tier responses are kept in
  `results/discarded/gemini_free_tier_key/` and are not scored. One reason for discarding them: the
  free-tier responses had compact JSON (`{"total_price":"91000",...}`), while the paid project's had
  a space after each colon (`{"total_price": "11,000", ...}`). Mixing the two inside one receipt's 5
  runs would have shown up as raw inconsistency that comes from the key, not from the model.

## 10. Running the calls

- **Retries:** only transient errors are retried (429, timeouts, connection errors, 5xx), with
  exponential backoff (2 s, 4 s, 8 s ... capped at 60 s, plus jitter), up to 8 attempts. The SDK's
  built-in retries are off so every attempt is visible and counted (`attempts` in each raw file).
  Gemini returned bursts of `503 "This model is currently experiencing high demand"` during setup.
  A 429 that says a per-day quota is exhausted fails at once instead, because waiting a minute
  cannot fix it.
- **Concurrency:** 4 calls at a time. Small enough not to trip rate limits, fast enough to finish
  250 calls in minutes.
- **Order:** run-major. Every receipt's run 1 goes before any run 2, so the five repeats of one
  request are spread over time rather than fired back to back.
- **Resume-safe:** each response is saved to `results/raw/{model}/{receipt_id}_run{k}.json` and
  existing files are skipped, so a crash or Ctrl-C loses nothing. Files are written to a temp name
  and renamed, so a half-written file can never be mistaken for a finished one. A failed call is
  not saved and is retried on the next run.
- **Raw files are committed** (about 4 MB) so anyone can re-score or audit without calling the APIs
  (`python run_eval.py --score-only`).
- **Secrets:** keys live only in `.env` (gitignored before the first commit) and are read with
  python-dotenv. They are never printed, and saved requests contain no headers except the
  cache-bypass header.

## Not done (on purpose)

No other tasks, datasets, models, temperatures, prompts or UI. No repeated runs on different days.
No prompt tuning. See the README's limitations section.
