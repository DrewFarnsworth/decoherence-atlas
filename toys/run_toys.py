"""Run the bridge toys and collect their flagged choices into toys/FLAGS.md.

    python toys/run_toys.py              all seven toys, full settings (toys 4 and 7 take longest)
    python toys/run_toys.py 1 2 6        just some
    python toys/run_toys.py --quick      quick modes (toys 4, 5, 7) and the fast toys
    python toys/run_toys.py --flags      only rebuild FLAGS.md from saved results
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bridge_lib as L  # noqa: E402

SCRIPTS = {"1": "toy1_markers.py", "2": "toy2_hom.py", "3": "toy3_superradiance.py", "4": "toy4_graph_domains.py",
           "5": "toy5_qed.py", "6": "toy6_radical_pair.py", "7": "toy7_parts.py"}
QUICKABLE = {"4", "5", "7"}

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--flags" not in args:
        quick = "--quick" in args
        which = [a for a in args if a in SCRIPTS] or list(SCRIPTS)
        for k in which:
            cmd = [sys.executable, str(HERE / SCRIPTS[k])] + (["--quick"] if quick and k in QUICKABLE else [])
            t = time.time()
            print(f"\n=== toy {k} ===", flush=True)
            r = subprocess.run(cmd)
            print(f"(toy {k}: exit {r.returncode}, {time.time() - t:.0f} s)")
    n = L.build_flags_md()
    print(f"\nFLAGS.md written: {n} flagged choices.")
