#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RequestResult:
    ok: bool
    latency_ms: float
    status_code: int
    error: str | None = None


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    rank = (len(sorted_values) - 1) * (p / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_values[lo]
    weight = rank - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def _send_one(url: str, payload: bytes, timeout: float) -> RequestResult:
    req = urllib.request.Request(
        url=url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            _ = resp.read()
            latency_ms = (time.perf_counter() - started) * 1000.0
            status = int(getattr(resp, "status", 200))
            return RequestResult(ok=(200 <= status < 300), latency_ms=latency_ms, status_code=status)
    except urllib.error.HTTPError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return RequestResult(ok=False, latency_ms=latency_ms, status_code=int(exc.code), error=str(exc))
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - started) * 1000.0
        return RequestResult(ok=False, latency_ms=latency_ms, status_code=0, error=type(exc).__name__)


def _run_load(
    *,
    url: str,
    payload: bytes,
    total: int,
    concurrency: int,
    timeout: float,
) -> tuple[list[RequestResult], float]:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_send_one, url, payload, timeout) for _ in range(total)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    elapsed_s = time.perf_counter() - started
    return results, elapsed_s


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark /chat concurrency for medical_agent API.")
    parser.add_argument("--url", default="http://127.0.0.1:8080/chat", help="Chat endpoint URL.")
    parser.add_argument("--question", default="左心衰竭时最早出现的症状是", help="Question payload.")
    parser.add_argument("--total", type=int, default=100, help="Total request count.")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent workers.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout (seconds).")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup requests (not counted).")
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write benchmark summary JSON.",
    )
    parser.add_argument(
        "--tag",
        default="baseline",
        help="Tag written in summary JSON, e.g. baseline / after-rerank-split.",
    )
    args = parser.parse_args()

    if args.total <= 0:
        raise SystemExit("--total must be > 0")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be > 0")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    payload = json.dumps({"question": args.question}, ensure_ascii=False).encode("utf-8")

    if args.warmup > 0:
        _ = _run_load(
            url=args.url,
            payload=payload,
            total=args.warmup,
            concurrency=min(args.concurrency, max(1, args.warmup)),
            timeout=args.timeout,
        )

    results, elapsed_s = _run_load(
        url=args.url,
        payload=payload,
        total=args.total,
        concurrency=args.concurrency,
        timeout=args.timeout,
    )

    latencies = sorted(r.latency_ms for r in results)
    success = sum(1 for r in results if r.ok)
    failures = len(results) - success
    qps = len(results) / elapsed_s if elapsed_s > 0 else 0.0

    summary = {
        "tag": args.tag,
        "url": args.url,
        "question": args.question,
        "total": len(results),
        "concurrency": args.concurrency,
        "elapsed_s": round(elapsed_s, 4),
        "qps": round(qps, 4),
        "success": success,
        "failures": failures,
        "error_rate": round((failures / len(results)) if results else 0.0, 6),
        "latency_ms": {
            "min": round(latencies[0], 3) if latencies else 0.0,
            "mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "p50": round(_percentile(latencies, 50), 3),
            "p95": round(_percentile(latencies, 95), 3),
            "p99": round(_percentile(latencies, 99), 3),
            "max": round(latencies[-1], 3) if latencies else 0.0,
        },
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

