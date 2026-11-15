"""One result format for every benchmark.

Each benchmark script builds a list of Check objects (computed value against
measured value) and an AtlasPoint (where the result sits on the atlas's
common axes), then calls write_result(). The runner reads the JSON files.

The pass rule is the project's: within 30% of the measured value, using the
paper's own inputs. For a measured range, within 30% of the nearer end. For a
measured bound ("at least", "at most"), the computed value must respect the
bound, with the same 30% allowance.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

TOLERANCE = 0.30


@dataclass
class Check:
    name: str                      # short description of the quantity
    unit: str
    computed: Optional[float]      # None when the model cannot produce it yet
    measured: Optional[float] = None
    measured_low: Optional[float] = None   # for ranges and lower bounds
    measured_high: Optional[float] = None  # for ranges and upper bounds
    source: str = ""
    note: str = ""
    scored: bool = True            # False for sensitivity runs and side results
    status: str = field(default="", init=False)
    ratio: Optional[float] = field(default=None, init=False)

    def evaluate(self) -> "Check":
        if self.computed is None or (isinstance(self.computed, float) and math.isnan(self.computed)):
            self.status = "not computed"
            return self
        x = self.computed
        if self.measured is not None:
            self.ratio = x / self.measured if self.measured else None
            ok = abs(x - self.measured) <= TOLERANCE * abs(self.measured)
        else:
            lo, hi = self.measured_low, self.measured_high
            ok = True
            if lo is not None and x < lo * (1 - TOLERANCE):
                ok = False
                self.ratio = x / lo
            if hi is not None and x > hi * (1 + TOLERANCE):
                ok = False
                self.ratio = x / hi
        self.status = "pass" if ok else "fail"
        if not self.scored:
            self.status = "info (" + self.status + ")"
        return self


@dataclass
class AtlasPoint:
    """Where a benchmark sits on the atlas's common axes.

    domain_size_m: the distance over which the event's records stay alike
        (record overlap above 1/e), or the spread of the undistinguished
        region (for a bound particle, its position spread).
    probe_wavelength_m: wavelength of the boson that carries the record.
    temperature_K: temperature of the system or its environment.
    time_s: the time over which the overlap was followed or decayed to 1/e.
    record_overlap: the computed I at that time (1 = indistinguishable).
    """
    label: str
    domain_size_m: Optional[float] = None
    probe_wavelength_m: Optional[float] = None
    temperature_K: Optional[float] = None
    time_s: Optional[float] = None
    record_overlap: Optional[float] = None
    boson: str = ""


def write_result(folder, benchmark: str, title: str, checks, points, notes=""):
    checks = [c.evaluate() for c in checks]
    scored = [c for c in checks if c.scored and c.status in ("pass", "fail")]
    if not scored:
        overall = "not computed"
    elif all(c.status == "pass" for c in scored):
        overall = "pass"
    elif any(c.status == "pass" for c in scored):
        overall = "partial"
    else:
        overall = "fail"
    if any(c.scored and c.status == "not computed" for c in checks) and overall == "pass":
        overall = "partial"
    data = {
        "benchmark": benchmark,
        "title": title,
        "overall": overall,
        "checks": [asdict(c) for c in checks],
        "atlas_points": [asdict(p) for p in points],
        "notes": notes,
    }
    path = Path(folder) / "result.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print_summary(data)
    return data


def _fmt(x):
    if x is None:
        return "-"
    if isinstance(x, (int, float)):
        if x == 0:
            return "0"
        if abs(x) >= 1e4 or abs(x) < 1e-2:
            return f"{x:.3g}"
        return f"{x:.3g}"
    return str(x)


def print_summary(data):
    print(f"\n{data['benchmark']}: {data['title']}  ->  {data['overall'].upper()}")
    for c in data["checks"]:
        if c["measured"] is not None:
            target = _fmt(c["measured"])
        else:
            lo, hi = c["measured_low"], c["measured_high"]
            target = f"{_fmt(lo) if lo is not None else ''}..{_fmt(hi) if hi is not None else ''}"
        print(f"  [{c['status']:>14}] {c['name']}: computed {_fmt(c['computed'])} {c['unit']}"
              f"  (measured {target} {c['unit']})")
