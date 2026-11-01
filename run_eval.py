"""Run the benchmark end to end: collect API responses, then score them.

    python run_eval.py --n 25 --runs 5

Step 1 calls each model `runs` times on each of the first `n` CORD test receipts and saves
every raw response to results/raw/{model}/{receipt_id}_run{k}.json. A call whose file
already exists is skipped, so an interrupted run can simply be started again (resume-safe).
Step 2 scores whatever is saved and writes results/summary.csv and results/per_run.csv.
"""

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

import metrics
from data import load_receipts
from extract import (MODELS, PRICES, SCHEMA, build_request, call_with_retries,
                     loggable_request, make_client)

RESULTS = Path(__file__).parent / "results"


def raw_path(model_name: str, receipt_id: str, k: int) -> Path:
    return RESULTS / "raw" / model_name / f"{receipt_id}_run{k}.json"


# --- Step 1: collect ------------------------------------------------------------------

def collect_one(client, model_name: str, receipt: dict, k: int) -> None:
    image = receipt["image_path"].read_bytes()
    request = build_request(model_name, image)
    result = call_with_retries(client, request)
    record = {
        "model_name": model_name,
        "receipt_id": receipt["id"],
        "run": k,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "request": loggable_request(request, image),  # every parameter sent, minus the image bytes
        **result,  # response (full), latency_s, attempts
    }
    path = raw_path(model_name, receipt["id"], k)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    tmp.rename(path)  # atomic, so a crash never leaves a half-written file that would be skipped


def collect(receipts: list[dict], n_runs: int, concurrency: int) -> None:
    clients = {m: make_client(m) for m in MODELS}
    # Run-major order: every receipt's run 1 before any run 2, so the repeats of one
    # request are spread out in time rather than fired back to back.
    jobs = [(m, r, k) for k in range(1, n_runs + 1) for r in receipts for m in MODELS
            if not raw_path(m, r["id"], k).exists()]
    print(f"Step 1: {len(jobs)} API calls to make ({concurrency} at a time)")
    failed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(collect_one, clients[m], m, r, k): (m, r["id"], k) for m, r, k in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            m, rid, k = futures[future]
            try:
                future.result()
                print(f"  [{i}/{len(jobs)}] {m} {rid} run{k} ok")
            except Exception as e:  # report and keep going; nothing is saved for this call
                failed += 1
                print(f"  [{i}/{len(jobs)}] {m} {rid} run{k} FAILED: {type(e).__name__}: {str(e)[:200]}")
    if failed:
        print(f"{failed} calls failed and were not saved. Run the same command again to retry them.")


# --- Step 2: score --------------------------------------------------------------------

def load_runs(model_name: str, receipts: list[dict], n_runs: int) -> dict[str, list[dict]]:
    runs = {}
    for receipt in receipts:
        runs[receipt["id"]] = []
        for k in range(1, n_runs + 1):
            path = raw_path(model_name, receipt["id"], k)
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            response = record["response"]
            choices = response.get("choices") or [{}]
            runs[receipt["id"]].append({
                "run": k,
                "content": (choices[0].get("message") or {}).get("content"),
                "latency_s": record["latency_s"],
                "usage": response.get("usage"),
                "vcache": response.get("vcache"),  # Interfaze only; should be False (cache bypassed)
            })
    return runs


def write_per_run_csv(all_runs: dict, gts: dict) -> None:
    """One row per (model, receipt, run): extracted values, ground truth and correctness."""
    rows = []
    for model_name, runs in all_runs.items():
        for rid, rs in runs.items():
            gt = gts[rid]
            for r in rs:
                pred = metrics.pred_fields(r["content"])
                row = {"model": model_name, "receipt_id": rid, "run": r["run"],
                       "schema_valid": metrics.is_schema_valid(r["content"], SCHEMA),
                       "latency_s": round(r["latency_s"], 2)}
                for field in metrics.FIELDS:
                    row[f"gt_{field}"] = json.dumps(gt[field], ensure_ascii=False)
                    row[f"out_{field}"] = json.dumps(pred[field] if pred else None, ensure_ascii=False)
                    for mode in ("strict", "lenient"):
                        row[f"{mode}_{field}"] = metrics.field_correct(pred, gt, field, mode)
                rows.append(row)
    with open(RESULTS / "per_run.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 0.1 else f"{value:.1f}"
    return str(value)


def score(receipts: list[dict], n_runs: int) -> None:
    gts = {r["id"]: metrics.gt_fields(r["gt_parse"]) for r in receipts}
    all_runs = {m: load_runs(m, receipts, n_runs) for m in MODELS}
    summaries = {m: metrics.summarise(all_runs[m], gts, SCHEMA, PRICES[m], n_runs) for m in MODELS}
    for m in MODELS:
        summaries[m] = {"model_id": MODELS[m]["model"], **summaries[m],
                        "vcache_hits": sum(bool(r["vcache"]) for rs in all_runs[m].values() for r in rs)}

    RESULTS.mkdir(exist_ok=True)
    metric_names = list(summaries[next(iter(MODELS))])
    with open(RESULTS / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", *MODELS])
        for name in metric_names:
            writer.writerow([name, *(fmt(summaries[m][name]) for m in MODELS)])
    write_per_run_csv(all_runs, gts)

    print(f"\nStep 2: scored {len(receipts)} receipts x {n_runs} runs")
    print(f"{'metric':<36}" + "".join(f"{m:>18}" for m in MODELS))
    for name in metric_names:
        print(f"{name:<36}" + "".join(f"{fmt(summaries[m][name]):>18}" for m in MODELS))
    print(f"\nWrote {RESULTS / 'summary.csv'} and {RESULTS / 'per_run.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=25, help="number of CORD test receipts (first n)")
    parser.add_argument("--runs", type=int, default=5, help="calls per receipt per model")
    parser.add_argument("--concurrency", type=int, default=4, help="parallel API calls")
    parser.add_argument("--score-only", action="store_true", help="skip API calls, score saved responses")
    args = parser.parse_args()

    load_dotenv(Path(__file__).parent / ".env")  # API keys; never printed or saved
    receipts = load_receipts(args.n)
    if not args.score_only:
        collect(receipts, args.runs, args.concurrency)
    score(receipts, args.runs)


if __name__ == "__main__":
    main()
