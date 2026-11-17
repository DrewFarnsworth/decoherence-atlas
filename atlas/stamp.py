"""Skip runs whose code and settings have not changed.

A run's fingerprint is a SHA-256 of the source files it depends on plus its settings. It is saved
as a .stamp file beside the run's results after a successful run. If the fingerprint matches next
time and the results are still there, the runner skips the run and keeps the old results.
Use --force on either runner to rerun anyway.
"""

import hashlib
import os
from pathlib import Path


def fingerprint(files, settings=""):
    h = hashlib.sha256()
    for f in sorted(Path(p) for p in files):
        h.update(str(f.name).encode())
        h.update(Path(f).read_bytes())
    h.update(str(settings).encode())
    return h.hexdigest()


def is_fresh(stamp_path, fp, must_exist=()):
    p = Path(stamp_path)
    if not p.exists() or any(not Path(m).exists() for m in must_exist):
        return False
    return p.read_text(encoding="utf-8").strip() == fp


def write_stamp(stamp_path, fp):
    Path(stamp_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stamp_path).write_text(fp + "\n", encoding="utf-8")


def workers():
    """Worker processes for parallel sweeps: every core, or ATLAS_WORKERS if set."""
    env = os.environ.get("ATLAS_WORKERS")
    return max(1, int(env)) if env else (os.cpu_count() or 1)
