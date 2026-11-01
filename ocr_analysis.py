"""Look inside Interfaze's OCR step, using the saved responses only (no API calls).

    python ocr_analysis.py

Every Interfaze response carries the output of its OCR pre-pass in `precontext`: the text it
read, plus each word's position and confidence. This script checks two things:

1. Is the OCR step stable across the 5 runs, and is the final answer stable?
2. How confident was the OCR on the words that ended up wrong in the final answer?
"""

import json
from pathlib import Path
from statistics import median

RAW = Path(__file__).parent / "results" / "raw" / "interfaze"
RECEIPTS = [f"cord_test_{i:03d}" for i in range(25)]
RUNS = range(1, 6)

# Misreads found by comparing Interfaze's answers with the ground truth and the images:
# (receipt, a word on the same OCR line, the word as the OCR read it, what the receipt shows,
#  skip the line's first word: the leading "1" on the JASMINE line is the quantity, not the misread)
MISREADS = [
    ("cord_test_002", "JASMINE", "1", "L  (item size, read as a digit)", True),
    ("cord_test_002", "28.000", "28.000", "28,000", False),
    ("cord_test_021", "TOTAL", "RP38,000", "Rp 38.000", False),
    ("cord_test_004", "AVOCADO", "67,000", "61,000  (a wrong reading the OCR was sure of)", False),
]


def load(receipt_id: str, k: int) -> dict:
    return json.loads((RAW / f"{receipt_id}_run{k}.json").read_text())["response"]


def ocr_text(response: dict) -> str:
    return response["precontext"][0]["result"]["extracted_text"]


def final_answer(response: dict) -> str:
    return response["choices"][0]["message"]["content"]


def ocr_lines(response: dict):
    """Each OCR line as (line text, [(word text, confidence), ...])."""
    for section in response["precontext"][0]["result"]["sections"]:
        for line in section["lines"]:
            yield line["text"], [(w["text"], w["confidence"]) for w in line["words"]]


def main() -> None:
    print("1. Stability: OCR text vs final answer, across 5 runs per receipt")
    ocr_stable = answer_stable = 0
    for rid in RECEIPTS:
        responses = [load(rid, k) for k in RUNS]
        same_ocr = len({ocr_text(r) for r in responses}) == 1
        same_answer = len({final_answer(r) for r in responses}) == 1
        ocr_stable += same_ocr
        answer_stable += same_answer
        if not (same_ocr and same_answer):
            print(f"   {rid}: OCR identical = {same_ocr}, final answer identical = {same_answer}")
    print(f"   OCR text identical in all 5 runs: {ocr_stable}/25 receipts")
    print(f"   Final answer identical in all 5 runs: {answer_stable}/25 receipts")

    print("\n2. OCR confidence on misread words")
    every_word = [c for rid in RECEIPTS for _, words in ocr_lines(load(rid, 1)) for _, c in words]
    print(f"   All {len(every_word)} words (run 1): median confidence {median(every_word):.2f}, "
          f"{sum(c < 0.8 for c in every_word) / len(every_word):.0%} below 0.80")
    for rid, line_has, read_as, printed, skip_first in MISREADS:
        found = sorted({c for k in RUNS for line, words in ocr_lines(load(rid, k)) if line_has in line
                        for word, c in (words[1:] if skip_first else words) if word == read_as})
        values = ", ".join(f"{c:.2f}" for c in found) or "not found"
        print(f"   {rid}: read '{read_as}', receipt shows {printed} -> confidence {values}")


if __name__ == "__main__":
    main()
