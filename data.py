"""Load the first N receipts of the CORD v2 test split and cache them under data/.

Each cached receipt is two files:
    data/cord_test/{receipt_id}.jpg   original image bytes (not re-encoded)
    data/cord_test/{receipt_id}.json  the sample's gt_parse (ground truth)

The cache makes runs reproducible and avoids re-downloading. data/ is gitignored.
"""

import json
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "data" / "cord_test"
DATASET = "naver-clova-ix/cord-v2"


def receipt_id(index: int) -> str:
    return f"cord_test_{index:03d}"


def _download(n: int) -> None:
    # Imported lazily so the metrics/tests never need the heavy `datasets` package.
    from datasets import Image, load_dataset

    ds = load_dataset(DATASET, split="test")
    # decode=False gives us the original JPEG bytes instead of a re-encoded PIL image,
    # so both models see exactly the file that was annotated.
    ds = ds.cast_column("image", Image(decode=False))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(n):  # the first n samples, in dataset order: fixed and reproducible
        sample = ds[i]
        rid = receipt_id(i)
        (CACHE_DIR / f"{rid}.jpg").write_bytes(sample["image"]["bytes"])
        gt_parse = json.loads(sample["ground_truth"])["gt_parse"]
        (CACHE_DIR / f"{rid}.json").write_text(json.dumps(gt_parse, ensure_ascii=False, indent=2))


def load_receipts(n: int) -> list[dict]:
    """Return [{"id", "image_path", "gt_parse"}] for the first n test receipts."""
    if not all((CACHE_DIR / f"{receipt_id(i)}.json").exists() for i in range(n)):
        _download(n)
    receipts = []
    for i in range(n):
        rid = receipt_id(i)
        receipts.append({
            "id": rid,
            "image_path": CACHE_DIR / f"{rid}.jpg",
            "gt_parse": json.loads((CACHE_DIR / f"{rid}.json").read_text()),
        })
    return receipts


if __name__ == "__main__":
    for r in load_receipts(25):
        print(r["id"], r["image_path"].stat().st_size, "bytes")
