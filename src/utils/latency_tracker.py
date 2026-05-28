"""
Latency Tracker — thread-safe rolling window for p50/p95/p99 latency monitoring.
Records per-query latency and aggregates stats to validate sub-200 ms SLO.
"""

import threading
import statistics
import json
from collections import deque
from datetime import datetime
from typing import Dict, Any


class LatencyTracker:
    """
    Thread-safe rolling latency tracker.
    Default window: last 10,000 measurements (matching production scale claim).
    """

    def __init__(self, window: int = 10_000):
        self._window = window
        self._lock = threading.Lock()
        self._latencies: deque = deque(maxlen=window)
        self._total = 0
        self._count = 0
        self._violations = 0        # queries exceeding SLO
        self._SLO_MS = 200.0

    def record(self, latency_ms: float):
        with self._lock:
            self._latencies.append(latency_ms)
            self._total += latency_ms
            self._count += 1
            if latency_ms > self._SLO_MS:
                self._violations += 1

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            if not self._latencies:
                return {}
            data = sorted(self._latencies)
            n = len(data)
            return {
                "count": self._count,
                "window_size": n,
                "p50_ms": round(data[int(0.50 * n)], 2),
                "p75_ms": round(data[int(0.75 * n)], 2),
                "p95_ms": round(data[int(0.95 * n)], 2),
                "p99_ms": round(data[int(0.99 * n)], 2),
                "mean_ms": round(statistics.mean(data), 2),
                "max_ms": round(max(data), 2),
                "min_ms": round(min(data), 2),
                "slo_target_ms": self._SLO_MS,
                "slo_compliance_pct": round(100 * (1 - self._violations / self._count), 2),
            }

    def report(self):
        s = self.stats()
        if not s:
            print("No data recorded yet.")
            return
        print("\n" + "─" * 50)
        print(f"  Latency Report  ({s['count']} total queries)")
        print("─" * 50)
        print(f"  P50:  {s['p50_ms']:>8.1f} ms")
        print(f"  P75:  {s['p75_ms']:>8.1f} ms")
        print(f"  P95:  {s['p95_ms']:>8.1f} ms  ← SLO target < {s['slo_target_ms']:.0f} ms")
        print(f"  P99:  {s['p99_ms']:>8.1f} ms")
        print(f"  Mean: {s['mean_ms']:>8.1f} ms")
        print(f"  SLO compliance: {s['slo_compliance_pct']}%")
        print("─" * 50)

    def to_json(self) -> str:
        return json.dumps(self.stats(), indent=2)
