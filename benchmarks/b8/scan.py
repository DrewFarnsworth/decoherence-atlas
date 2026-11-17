"""B8 fix: is there one Drude-Lorentz bath that gives both measured coherence times?

Target (Duan 2017): about 60 fs for electronic coherence at room temperature. The scored rows are
the optical dephasing time and the inter-exciton coherence time (HEOM). The standard bath
(lambda = 35 cm^-1, 1/gamma = 50 fs) gives 78 and 148 fs; lambda = 190 gives 29 and 54 fs.

    python benchmarks/b8/scan.py            optical dephasing over a grid of (lambda, 1/gamma): about a minute
    python benchmarks/b8/scan.py --heom     adds HEOM at the grid points within 30% on optical dephasing (slow: hours)

Writes benchmarks/b8/scan.md. Reports which baths, if any, put both times within 30% of 60 fs.
This is a scan, not a fit: it reports every grid point and does not tune anything to the answer.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("b8run", HERE / "run.py")
b8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b8)

LAMBDAS = [20, 35, 50, 70, 100, 140, 190, 260]
TAUS = [25, 50, 100, 200]            # 1/gamma in fs
TARGET, TOL, T = 60.0, 0.30, 294.0


def main(heom=False):
    rows = []
    for lam in LAMBDAS:
        for tau in TAUS:
            gcm = 1 / (tau * b8.W)
            opt = b8.optical_dephasing_time(T, lam=lam, gamma_cm=gcm)
            ok_opt = abs(opt - TARGET) <= TOL * TARGET
            exc = float("nan")
            if heom and ok_opt:
                old = b8.TAU_C
                b8.TAU_C = tau
                exc, _, _ = b8.exciton_coherence_time(T, lam=lam, Nk=1, depth=5 if lam > 100 else 4)
                b8.TAU_C = old
            both = ok_opt and abs(exc - TARGET) <= TOL * TARGET
            rows.append((lam, tau, opt, exc, ok_opt, both))
            print(f"lambda {lam:4d} cm-1, 1/gamma {tau:4d} fs: optical {opt:6.1f} fs, exciton {exc:6.1f} fs"
                  f"{'  <- optical within 30%' if ok_opt else ''}{'  <- BOTH' if both else ''}", flush=True)
    lines = ["# B8 bath scan", "", f"Room temperature ({T:g} K); target {TARGET:g} fs, within 30%. Drude-Lorentz bath on every pigment.", "",
             "| lambda (cm^-1) | 1/gamma (fs) | optical dephasing (fs) | inter-exciton coherence (fs) | optical within 30% | both within 30% |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {a} | {b} | {c:.1f} | {'–' if d != d else f'{d:.1f}'} | {e} | {f} |" for a, b, c, d, e, f in rows]
    found = [r for r in rows if r[5]]
    lines += ["", ("Baths giving both: " + ", ".join(f"lambda {r[0]}, 1/gamma {r[1]} fs" for r in found)) if found
              else ("No bath on this grid gives both." if heom else "Run with --heom to test the inter-exciton time.")]
    (HERE / "scan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(heom="--heom" in sys.argv)
