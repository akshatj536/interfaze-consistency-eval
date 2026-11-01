"""Scoring. Every function here is pure (no I/O, no API calls) and has a test in tests/.

Vocabulary
- "fields": a dict with the 4 evaluated fields, built the same way from ground truth
  and from model output: total_price, subtotal_price, item_count, item_names.
- STRICT scoring: exact string match after trimming outer whitespace. Measures OCR fidelity.
- LENIENT scoring: prices reduced to digits, names lowercased + trimmed. Measures value correctness.
- A "run" is one API call's result: {"content": str | None, "latency_s": float, "usage": dict | None}.
"""

import json
import math
import re

import jsonschema

FIELDS = ["total_price", "subtotal_price", "item_count", "item_names"]
INVALID = "<invalid output>"  # stands in for any field of an output that isn't a JSON object


# --- Building the 4 fields --------------------------------------------------------

def _first_if_list(value):
    # CORD stores a value annotated twice on the receipt as a list (e.g. one subtotal
    # printed twice: ["46.636", "46.636"]). We take the first annotation.
    return value[0] if isinstance(value, list) and value else value


def gt_fields(gt_parse: dict) -> dict:
    """The 4 fields from a CORD gt_parse. A missing price is None (that field is skipped)."""
    menu = gt_parse.get("menu") or []
    if isinstance(menu, dict):  # CORD stores a single-item menu as a dict, not a list
        menu = [menu]
    return {
        "total_price": _first_if_list((gt_parse.get("total") or {}).get("total_price")),
        "subtotal_price": _first_if_list((gt_parse.get("sub_total") or {}).get("subtotal_price")),
        "item_count": len(menu),
        "item_names": [m.get("nm") for m in menu],
    }


def parse_output(content: str | None) -> dict | None:
    """The model's message content as a dict, or None if it is not a JSON object."""
    try:
        obj = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def is_schema_valid(content: str | None, schema: dict) -> bool:
    obj = parse_output(content)
    if obj is None:
        return False
    try:
        jsonschema.validate(obj, schema)
        return True
    except jsonschema.ValidationError:
        return False


def pred_fields(content: str | None) -> dict | None:
    """The 4 fields from a model's output, or None if the output is not a JSON object."""
    obj = parse_output(content)
    if obj is None:
        return None
    items = obj.get("items") if isinstance(obj.get("items"), list) else []
    return {
        "total_price": obj.get("total_price"),
        "subtotal_price": obj.get("subtotal_price"),
        "item_count": len(items),
        "item_names": [i.get("name") if isinstance(i, dict) else None for i in items],
    }


# --- Normalisation ------------------------------------------------------------------

def strict_text(value) -> str | None:
    """STRICT: trim outer whitespace only. Non-strings (e.g. null) become None."""
    return value.strip() if isinstance(value, str) else None


def lenient_price(value) -> int | None:
    """LENIENT price: keep digits only. "45,500" -> 45500, "Rp 45.500" -> 45500.

    Valid here because every CORD price is in whole rupiah (no decimal part), so dots
    and commas are always thousands separators. Not valid for currencies with cents.
    """
    if not isinstance(value, str):
        return None
    digits = re.sub(r"\D", "", value)
    return int(digits) if digits else None


def lenient_name(value) -> str | None:
    """LENIENT name: lowercase and trim outer whitespace."""
    return value.strip().lower() if isinstance(value, str) else None


def field_value(fields: dict, field: str, mode: str):
    """The value used for comparison of one field, under "strict" or "lenient" scoring."""
    value = fields[field]
    if field == "item_count":
        return value  # an integer: strict and lenient are the same
    if field == "item_names":
        normalise = strict_text if mode == "strict" else lenient_name
        # A sorted tuple compares equal exactly when the two lists are equal as multisets.
        return tuple(sorted((normalise(n) or "") for n in value))
    return strict_text(value) if mode == "strict" else lenient_price(value)


def raw_field_value(fields: dict | None, field: str):
    """A field exactly as the model returned it (no trimming, item order kept). For raw consistency."""
    if fields is None:
        return INVALID
    value = fields[field]
    return tuple(value) if isinstance(value, list) else value


# --- Accuracy -------------------------------------------------------------------------

def field_correct(pred: dict | None, gt: dict, field: str, mode: str) -> bool | None:
    """Whether one field is correct. None means the ground truth lacks it, so it is skipped."""
    if gt[field] is None:
        return None
    if pred is None:  # unparseable output: every field counts as wrong
        return False
    return field_value(pred, field, mode) == field_value(gt, field, mode)


def all_fields_correct(pred: dict | None, gt: dict, mode: str) -> bool:
    """True when every field that has ground truth is correct."""
    results = [field_correct(pred, gt, f, mode) for f in FIELDS]
    return all(r for r in results if r is not None)


def pct(flags: list) -> float | None:
    """Percentage of True among the flags, ignoring None (skipped). None if nothing to count."""
    counted = [f for f in flags if f is not None]
    return 100 * sum(counted) / len(counted) if counted else None


# --- Consistency -----------------------------------------------------------------------

def normalised_output(pred: dict | None) -> tuple:
    """A whole output after LENIENT normalisation, as one comparable value."""
    if pred is None:
        return (INVALID,)
    return tuple(field_value(pred, f, "lenient") for f in FIELDS)


def n_distinct(values: list) -> int:
    return len(set(values))


def all_identical(values: list) -> bool:
    return n_distinct(values) == 1


# --- Latency and cost -------------------------------------------------------------------

def percentile(values: list[float], q: float) -> float:
    """The q-th percentile (0-100) with linear interpolation, like numpy's default."""
    s = sorted(values)
    k = (len(s) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def cost_usd(usage: dict | None, prices: dict) -> float | None:
    """Estimated cost of one call from reported token usage.

    Output is total_tokens - prompt_tokens rather than completion_tokens, because Gemini
    leaves its (billed) thinking tokens out of completion_tokens but counts them in the total.
    """
    if not usage:
        return None
    input_tokens = usage["prompt_tokens"]
    output_tokens = usage["total_tokens"] - usage["prompt_tokens"]
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


# --- Putting it together -----------------------------------------------------------------

def summarise(runs: dict[str, list[dict]], gts: dict[str, dict], schema: dict,
              prices: dict, n_runs: int) -> dict:
    """All metrics for one model.

    runs: receipt_id -> list of runs for that receipt (missing calls simply absent)
    gts:  receipt_id -> gt_fields(...)
    Accuracy, validity, latency and cost use every run we have. Consistency uses only
    receipts that have all n_runs runs, so every receipt is judged on the same number of tries.
    """
    s = {}
    all_runs = [(rid, r) for rid, rs in runs.items() for r in rs]
    preds = [(rid, pred_fields(r["content"])) for rid, r in all_runs]
    s["calls"] = len(all_runs)
    s["schema_valid_%"] = pct([is_schema_valid(r["content"], schema) for _, r in all_runs])

    for mode in ("strict", "lenient"):
        for field in FIELDS:
            s[f"acc_{mode}_{field}_%"] = pct([field_correct(p, gts[rid], field, mode) for rid, p in preds])
        s[f"acc_{mode}_all_fields_%"] = pct([all_fields_correct(p, gts[rid], mode) for rid, p in preds])

    complete = {rid: rs for rid, rs in runs.items() if len(rs) == n_runs}
    s["receipts_with_all_runs"] = len(complete)
    raw_outputs = {rid: [r["content"] for r in rs] for rid, rs in complete.items()}
    parsed = {rid: [pred_fields(r["content"]) for r in rs] for rid, rs in complete.items()}
    norm_outputs = {rid: [normalised_output(p) for p in ps] for rid, ps in parsed.items()}

    for label, outputs in (("raw", raw_outputs), ("norm", norm_outputs)):
        s[f"consistency_{label}_exact_%"] = pct([all_identical(v) for v in outputs.values()])
        distinct = [n_distinct(v) for v in outputs.values()]
        s[f"distinct_{label}_mean"] = sum(distinct) / len(distinct) if distinct else None
        s[f"distinct_{label}_max"] = max(distinct) if distinct else None
    for field in FIELDS:
        s[f"agree_raw_{field}_%"] = pct([all_identical([raw_field_value(p, field) for p in ps])
                                         for ps in parsed.values()])
        s[f"agree_norm_{field}_%"] = pct([all_identical([INVALID if p is None else field_value(p, field, "lenient")
                                                         for p in ps]) for ps in parsed.values()])

    latencies = [r["latency_s"] for _, r in all_runs]
    s["latency_median_s"] = percentile(latencies, 50) if latencies else None
    s["latency_p90_s"] = percentile(latencies, 90) if latencies else None
    costs = [cost_usd(r.get("usage"), prices) for _, r in all_runs]
    known = [c for c in costs if c is not None]
    s["cost_per_call_usd"] = sum(known) / len(known) if known else None
    s["cost_total_usd"] = sum(known) if known else None
    return s
