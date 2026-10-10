"""Per-step timing. Every pipeline step is wrapped in ``timer.step(name)`` so runs log real durations."""

from __future__ import annotations

import logging
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

log = logging.getLogger("avtopassport.timing")


@dataclass
class StepTimer:
    rows: list[dict[str, Any]] = field(default_factory=list)

    @contextmanager
    def step(self, name: str, **meta: Any) -> Iterator[None]:
        t0 = time.perf_counter()
        ok = True
        try:
            yield
        except Exception:
            ok = False
            raise
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            self.rows.append({"step": name, "ms": ms, "ok": ok, **meta})
            log.debug("step %s took %.1f ms", name, ms)

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, list[float]] = {}
        for r in self.rows:
            out.setdefault(r["step"], []).append(r["ms"])
        return {
            k: {"count": len(v), "total_ms": sum(v), "median_ms": statistics.median(v), "max_ms": max(v)}
            for k, v in out.items()
        }
