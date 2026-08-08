"""
Measure end-to-end pipeline latency against the running gateway.
Reports P50/P95/P99 to compare with the proposal's <800ms target.

Usage: python scripts/benchmark_latency.py --url http://localhost:8000 --n 50 --wav sample.wav
"""
import argparse
import base64
import statistics
import time

import httpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--wav", required=True, help="path to a wav file to send")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--source", default="auto")
    ap.add_argument("--target", default="es")
    args = ap.parse_args()

    audio_b64 = base64.b64encode(open(args.wav, "rb").read()).decode()
    payload = {"audio_b64": audio_b64, "source": args.source, "target": args.target}

    latencies = []
    with httpx.Client(timeout=60) as c:
        for i in range(args.n):
            t0 = time.perf_counter()
            r = c.post(f"{args.url}/translate", json=payload)
            r.raise_for_status()
            latencies.append((time.perf_counter() - t0) * 1000)
            print(f"[{i+1}/{args.n}] {latencies[-1]:.0f} ms", end="\r")

    latencies.sort()
    def p(q):
        return latencies[min(len(latencies) - 1, int(len(latencies) * q))]
    print("\n--- end-to-end latency (ms) ---")
    print(f"  P50 {p(0.50):.0f}   P95 {p(0.95):.0f}   P99 {p(0.99):.0f}")
    print(f"  mean {statistics.mean(latencies):.0f}   min {latencies[0]:.0f}   max {latencies[-1]:.0f}")
    print(f"  target < 800 ms -> {'PASS' if p(0.95) < 800 else 'FAIL'} (on P95)")


if __name__ == "__main__":
    main()
