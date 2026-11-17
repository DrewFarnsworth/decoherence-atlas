"""Run the bridge toys and collect their flagged choices into toys/FLAGS.md.

    python toys/run_toys.py              all seven toys, full settings (toys 4 and 7 take longest)
    python toys/run_toys.py 1 2 6        just some
    python toys/run_toys.py --quick      quick modes (toys 4, 5, 7) and the fast toys
    python toys/run_toys.py --flags      only rebuild FLAGS.md from saved results
    python toys/run_toys.py --force      rerun even toys whose code has not changed

A toy whose script, the shared library and any toy it imports are unchanged since its last
successful run (in the same mode) is skipped, and its saved results are kept.
Sweeps use every CPU core; set ATLAS_WORKERS to use fewer.
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import bridge_lib as L  # noqa: E402
from atlas.stamp import fingerprint, is_fresh, write_stamp  # noqa: E402

SCRIPTS = {"1": "toy1_markers.py", "2": "toy2_hom.py", "3": "toy3_superradiance.py", "4": "toy4_graph_domains.py",
           "5": "toy5_qed.py", "6": "toy6_radical_pair.py", "7": "toy7_parts.py"}
QUICKABLE = {"4", "5", "7"}
DEPENDS = {"3": ["toy2_hom.py"], "7": ["toy4_graph_domains.py"]}   # other toy scripts each toy imports

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--flags" not in args:
        quick = "--quick" in args
        which = [a for a in args if a in SCRIPTS] or list(SCRIPTS)
        force = "--force" in args
        for k in which:
            mode = "--quick" if quick and k in QUICKABLE else ""
            cmd = [sys.executable, str(HERE / SCRIPTS[k])] + ([mode] if mode else [])
            files = [HERE / SCRIPTS[k], HERE / "bridge_lib.py"] + [HERE / d for d in DEPENDS.get(k, [])]
            fp = fingerprint(files, settings=mode)
            outdir = L.RESULTS / f"toy{k}"
            stamp = outdir / ".stamp"
            if not force and is_fresh(stamp, fp, must_exist=[outdir / "SUMMARY.md"]):
                print(f"\n=== toy {k} === unchanged since its last run{' (quick)' if mode else ''}: skipped (use --force to rerun)")
                continue
            t = time.time()
            print(f"\n=== toy {k} ===", flush=True)
            r = subprocess.run(cmd)
            print(f"(toy {k}: exit {r.returncode}, {time.time() - t:.0f} s)")
            if r.returncode == 0:
                write_stamp(stamp, fp)
    n = L.build_flags_md()
    print(f"\nFLAGS.md written: {n} flagged choices.")
