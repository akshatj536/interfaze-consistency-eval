import json

import pytest

from extract import PRICES, SCHEMA
from metrics import (
    INVALID, all_fields_correct, all_identical, cost_usd, field_correct, field_value,
    gt_fields, is_schema_valid, lenient_name, lenient_price, n_distinct, normalised_output,
    pct, percentile, pred_fields, raw_field_value, strict_text, summarise,
)


def output(total="45,500", subtotal=None, names=("A",)):
    """A model output string in the requested schema."""
    return json.dumps({"total_price": total, "subtotal_price": subtotal,
                       "items": [{"name": n} for n in names]})


# --- Building fields -------------------------------------------------------------------

def test_gt_fields_single_item_menu_is_a_dict():
    gt = gt_fields({"menu": {"nm": "TEA"}, "total": {"total_price": "5,000"}})
    assert gt == {"total_price": "5,000", "subtotal_price": None, "item_count": 1, "item_names": ["TEA"]}


def test_gt_fields_takes_first_of_a_doubly_annotated_value():
    gt = gt_fields({"menu": [], "sub_total": {"subtotal_price": ["46.636", "46.636"]}})
    assert gt["subtotal_price"] == "46.636"
    assert gt["total_price"] is None


def test_pred_fields_reads_output_and_rejects_non_objects():
    assert pred_fields(output(names=["A", "B"])) == {
        "total_price": "45,500", "subtotal_price": None, "item_count": 2, "item_names": ["A", "B"]}
    assert pred_fields("not json") is None
    assert pred_fields("[1, 2]") is None
    assert pred_fields(None) is None


def test_schema_validity():
    assert is_schema_valid(output(), SCHEMA)
    assert is_schema_valid(output(subtotal="40,000"), SCHEMA)
    assert not is_schema_valid('{"total_price": "1"}', SCHEMA)  # missing keys
    assert not is_schema_valid(output()[:-1] + ', "extra": 1}', SCHEMA)  # extra key
    assert not is_schema_valid("```json\n" + output() + "\n```", SCHEMA)  # fenced, not raw JSON


# --- Normalisation ----------------------------------------------------------------------

def test_strict_text_trims_outer_whitespace_only():
    assert strict_text("  JASMINE MT ( L ) ") == "JASMINE MT ( L )"
    assert strict_text(None) is None


@pytest.mark.parametrize("text, expected", [
    ("45,500", 45500), ("Rp 45.500", 45500), ("Rp.118.000", 118000), ("91000", 91000),
    ("abc", None), (None, None),
])
def test_lenient_price_keeps_digits_only(text, expected):
    assert lenient_price(text) == expected


def test_lenient_name():
    assert lenient_name("  Ice Tea ") == "ice tea"
    assert lenient_name(None) is None


def test_item_names_compare_as_multiset():
    a = {"item_names": ["B", "A", "A"]}
    assert field_value(a, "item_names", "strict") == field_value({"item_names": ["A", "B", "A"]}, "item_names", "strict")
    assert field_value(a, "item_names", "strict") != field_value({"item_names": ["A", "B"]}, "item_names", "strict")
    assert field_value({"item_names": ["Tea "]}, "item_names", "lenient") == ("tea",)


# --- Accuracy -----------------------------------------------------------------------------

def test_strict_vs_lenient_on_a_price():
    gt = {"total_price": "38.000", "subtotal_price": None, "item_count": 1, "item_names": ["A"]}
    pred = pred_fields(output(total="Rp 38,000"))
    assert field_correct(pred, gt, "total_price", "strict") is False
    assert field_correct(pred, gt, "total_price", "lenient") is True


def test_missing_ground_truth_is_skipped_and_invalid_output_is_wrong():
    gt = {"total_price": "1", "subtotal_price": None, "item_count": 1, "item_names": ["A"]}
    assert field_correct(pred_fields(output(subtotal="9")), gt, "subtotal_price", "strict") is None
    assert field_correct(None, gt, "total_price", "strict") is False


def test_all_fields_correct_ignores_skipped_fields():
    gt = {"total_price": "45,500", "subtotal_price": None, "item_count": 1, "item_names": ["A"]}
    assert all_fields_correct(pred_fields(output(subtotal="anything")), gt, "strict")
    assert not all_fields_correct(pred_fields(output(names=["A", "B"])), gt, "strict")
    assert not all_fields_correct(None, gt, "strict")


def test_pct_ignores_none():
    assert pct([True, False, None, True]) == pytest.approx(200 / 3)
    assert pct([None]) is None


# --- Consistency ---------------------------------------------------------------------------

def test_raw_vs_normalised_consistency():
    a = pred_fields(output(total="45,500", names=["Tea"]))
    b = pred_fields(output(total="45.500", names=["TEA "]))
    assert normalised_output(a) == normalised_output(b)  # same values...
    assert raw_field_value(a, "total_price") != raw_field_value(b, "total_price")  # ...different text
    assert normalised_output(None) == (INVALID,)


def test_raw_field_value_keeps_item_order():
    a = pred_fields(output(names=["A", "B"]))
    b = pred_fields(output(names=["B", "A"]))
    assert raw_field_value(a, "item_names") != raw_field_value(b, "item_names")
    assert raw_field_value(None, "item_names") == INVALID


def test_identical_and_distinct():
    assert all_identical(["x", "x", "x"])
    assert not all_identical(["x", "y", "x"])
    assert n_distinct(["x", "y", "x"]) == 2


# --- Latency and cost -----------------------------------------------------------------------

def test_percentile_matches_numpy_linear():
    values = list(range(1, 11))
    assert percentile(values, 50) == pytest.approx(5.5)
    assert percentile(values, 90) == pytest.approx(9.1)
    assert percentile([3.0], 90) == 3.0


def test_cost_counts_hidden_thinking_tokens_as_output():
    # Gemini-style usage: completion_tokens (40) excludes ~1,108 thinking tokens.
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 40, "total_tokens": 2_000_000}
    assert cost_usd(usage, {"input": 1.0, "output": 2.0}) == pytest.approx(3.0)
    assert cost_usd(None, PRICES["gemini"]) is None


# --- End to end on fake data -------------------------------------------------------------------

def test_summarise_on_fake_runs():
    gts = {
        "r1": {"total_price": "45,500", "subtotal_price": None, "item_count": 1, "item_names": ["Tea"]},
        "r2": {"total_price": "10.000", "subtotal_price": "9.000", "item_count": 1, "item_names": ["Rice"]},
    }

    def run(content):
        return {"content": content, "latency_s": 1.0,
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}}

    runs = {
        # r1: 3 byte-identical, correct outputs
        "r1": [run(output("45,500", None, ["Tea"]))] * 3,
        # r2: same values written two ways -> raw differs, normalised agrees; strict fails once
        "r2": [run(output("10.000", "9.000", ["Rice"])), run(output("10,000", "9.000", ["Rice"])),
               run(output("10.000", "9.000", ["Rice"]))],
    }
    s = summarise(runs, gts, SCHEMA, {"input": 1.0, "output": 1.0}, n_runs=3)
    assert s["calls"] == 6
    assert s["schema_valid_%"] == 100
    assert s["acc_strict_total_price_%"] == pytest.approx(500 / 6)
    assert s["acc_lenient_total_price_%"] == 100
    assert s["acc_strict_subtotal_price_%"] == 100  # r1 skipped: only r2's 3 runs count
    assert s["consistency_raw_exact_%"] == 50
    assert s["consistency_norm_exact_%"] == 100
    assert s["distinct_raw_max"] == 2
    assert s["agree_raw_total_price_%"] == 50
    assert s["agree_norm_total_price_%"] == 100
    assert s["latency_median_s"] == 1.0
    assert s["cost_per_call_usd"] == pytest.approx(110 / 1_000_000)


def test_summarise_excludes_incomplete_receipts_from_consistency():
    gts = {"r1": {"total_price": "1", "subtotal_price": None, "item_count": 1, "item_names": ["A"]}}
    runs = {"r1": [{"content": output("1", None, ["A"]), "latency_s": 1.0, "usage": None}] * 2}
    s = summarise(runs, gts, SCHEMA, PRICES["gemini"], n_runs=3)
    assert s["receipts_with_all_runs"] == 0
    assert s["consistency_raw_exact_%"] is None
    assert s["acc_strict_total_price_%"] == 100
