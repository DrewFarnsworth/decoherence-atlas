"""Toy 4: domains on a small graph (Haken-Strobl-Reineker excitation transport).

N sites, one excitation, no decay. H = sum eps_k |k><k| + sum V_kl (|k><l| + h.c.);
local dephasing L_k = sqrt(gamma) |k><k|. Alternatives: the site a where the excitation was
absorbed (uniform prior); record rho_a(t) = state at time t after starting in |a>.

For every candidate set R (2 <= |R| <= N - 1) and two channels,
  Phi^trace: block of rho on R plus one 'outside' state;
  Phi^sum:   populations summed within each block (per graph) intersected with R, plus 'outside';
compute J^R for all pairs, sigma(R), m_i(R), and mark R as a domain under definition A (local
maximum of sigma) and B (self-consistency). Domains are outputs, not inputs.

Run: python toys/toy4_graph_domains.py            (full sweep)
     python toys/toy4_graph_domains.py --quick    (one point per graph, for tests)
"""

import csv
import itertools
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge_lib as L  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.stamp import workers  # noqa: E402

V_LIST = [0.0, 0.05, 0.2, 1.0]
W_LIST = [0.0, 0.5, 2.0]
G_LIST = [0.0, 0.1, 1.0, 10.0, 100.0]
T_LIST = [0.3, 1.0, 3.0, 10.0, 30.0]
G4_SEEDS = list(range(10))
DISORDER_SEED = 4040
J_FLOOR = 1e-6
HEADER = {"event": "an excitation is absorbed at one site and spreads", "alternatives": "the site a where it was absorbed",
          "prior": "uniform over sites"}


# ---------------------------------------------------------------- graphs
def graph(name, v=0.0, seed=0):
    """Return coupling matrix V (N x N) and the blocks for Phi^sum (None = no natural blocks)."""
    if name in ("G1", "G3"):
        N = 8
        V = np.zeros((N, N))
        for c in (range(0, 4), range(4, 8)):
            for i, j in itertools.combinations(c, 2):
                V[i, j] = V[j, i] = 1.0
        if name == "G3":
            V[0, 1] = V[1, 0] = -1.0          # one negative bond on the triangle 0-1-2 of cluster A
        V[3, 4] = V[4, 3] = v                 # the single link between the clusters
        return V, [list(range(0, 4)), list(range(4, 8))]
    if name == "G2":
        N = 8
        V = np.zeros((N, N))
        for i in range(N - 1):
            V[i, i + 1] = V[i + 1, i] = 1.0
        return V, [[0, 1], [2, 3], [4, 5], [6, 7]]
    if name == "G4":
        N = 10
        rng = np.random.default_rng(seed)
        V = np.zeros((N, N))
        for i, j in itertools.combinations(range(N), 2):
            if rng.random() < 0.4:
                V[i, j] = V[j, i] = rng.uniform(-1, 1)
        return V, None
    raise ValueError(name)


def disorder(N, W, seed=DISORDER_SEED):
    return np.random.default_rng(seed + N).normal(0, W, N) if W > 0 else np.zeros(N)


# ---------------------------------------------------------------- dynamics
def liouvillian(H, gamma):
    N = H.shape[0]
    I = np.eye(N)
    Lsup = -1j * (np.kron(I, H) - np.kron(H.T, I))
    off = (np.ones((N, N)) - np.eye(N)).flatten(order="F")
    return Lsup - gamma * np.diag(off)


def records(V, eps, gamma, t):
    """rho_a(t) for every starting site a; gamma = inf means records stay |a><a|."""
    N = V.shape[0]
    if math.isinf(gamma):
        return np.array([np.diag(np.eye(N)[a]).astype(complex) for a in range(N)])
    H = np.diag(eps) + V
    P = expm(liouvillian(H, gamma) * t)
    recs = []
    for a in range(N):
        r0 = np.zeros((N, N), complex); r0[a, a] = 1
        r = (P @ r0.flatten(order="F")).reshape(N, N, order="F")
        L.check(abs(np.trace(r).real - 1) < 1e-10, "trace preserved")
        recs.append((r + r.conj().T) / 2)
    return np.array(recs)


# ---------------------------------------------------------------- channels (batched over records)
def reg_trace(recs, R):
    idx = sorted(R)
    n, d = recs.shape[0], len(idx) + 1
    out = np.zeros((n, d, d), complex)
    out[:, :-1, :-1] = recs[:, idx][:, :, idx]
    out[:, -1, -1] = np.clip(1 - np.real(np.trace(out[:, :-1, :-1], axis1=1, axis2=2)), 0, None)
    return out


def reg_sum(recs, R, blocks):
    pops = np.real(np.einsum("aii->ai", recs))
    parts = [[i for i in b if i in R] for b in blocks]
    parts = [p for p in parts if p]
    cols = [pops[:, p].sum(axis=1) for p in parts]
    inside = np.stack(cols, axis=1)
    outside = np.clip(1 - inside.sum(axis=1), 0, None)
    return np.concatenate([inside, outside[:, None]], axis=1)


def J_quantum(states):
    """All-pairs root fidelity for a stack of density matrices (n, d, d)."""
    S = L.psd_sqrt(states)
    M = np.einsum("aij,bjk,akl->abil", S, states, S)
    M = (M + M.conj().swapaxes(-1, -2)) / 2
    w = np.linalg.eigvalsh(M)
    J = np.sum(np.sqrt(np.clip(w, 0, None)), axis=-1)
    J[J < J_FLOOR] = 0.0                      # below the numerical floor of the root fidelity
    np.fill_diagonal(J, 1.0)
    return np.minimum(J, 1.0)


def J_classical(p):
    J = np.sqrt(np.clip(p, 0, None)) @ np.sqrt(np.clip(p, 0, None)).T
    np.fill_diagonal(J, 1.0)
    return np.minimum(J, 1.0)


# ---------------------------------------------------------------- domain search at one point
def analyse(recs, blocks, channels):
    N = recs.shape[0]
    subs = list(L.subsets(N))
    res = {}
    for ch in channels:
        Jall, sig, mem = {}, {}, {}
        for R in subs:
            if ch == "trace":
                J = J_quantum(reg_trace(recs, R))
            else:
                J = J_classical(reg_sum(recs, R, blocks))
            Jall[R] = J
            sig[R] = L.sigma_of(J, R)
        domA = [R for R in subs if L.is_domain_A(lambda S: sig.get(S, float("nan")), R, N)]
        domB = [R for R in subs if L.is_domain_B(Jall[R], R)]
        res[ch] = {"J": Jall, "sigma": sig, "A": domA, "B": domB}
    return res


def relation(R1, R2):
    if R1 == R2:
        return "same"
    if R1 < R2:
        return "inside"
    if R2 < R1:
        return "contains"
    return "overlap" if R1 & R2 else "disjoint"


def fmt_set(R):
    return "{" + ",".join(str(i) for i in sorted(R)) + "}"


def point_job(args):
    gname, v, seed, W, gamma, t, channels = args
    V, blocks = graph(gname, v, seed)
    N = V.shape[0]
    eps = disorder(N, W)
    recs = records(V, eps, gamma, t)
    res = analyse(recs, blocks, channels)
    out = {"key": (gname, v, seed, W, gamma, t), "domains": [], "milestone": [], "disagree": [], "strongest": {}}
    for ch in channels:
        r = res[ch]
        for defn in ("A", "B"):
            for R in sorted(r[defn], key=lambda S: -r["sigma"][S]):
                out["domains"].append([ch, defn, fmt_set(R), len(R), r["sigma"][R]])
            out["strongest"][(ch, defn)] = max((r["sigma"][R] for R in r[defn]), default=float("nan"))
        a, b = set(r["A"]), set(r["B"])
        for R in sorted(a ^ b, key=lambda S: (len(S), sorted(S))):
            out["disagree"].append([ch, fmt_set(R), "A only" if R in a else "B only", r["sigma"][R]])
    # monotonicity checks (sampled): smaller R' registers less, so joins more; sum joins more than trace
    if "trace" in channels:
        Jt = res["trace"]["J"]
        for R in list(Jt)[::17]:
            for k in R:
                R2 = R - {k}
                if len(R2) >= 2:
                    L.check(np.all(Jt[R2] >= Jt[R] - 1e-8), f"monotonicity under partial trace {fmt_set(R2)} in {fmt_set(R)}")
            if "sum" in channels:
                L.check(np.all(res["sum"]["J"][R] >= Jt[R] - 1e-8), "Phi^sum joins at least as much as Phi^trace")
    # milestone test: domains under Phi^sum with pairs cut at a finer (trace) grain
    if "sum" in channels and "trace" in channels:
        Jt, Js = res["trace"]["J"], res["sum"]["J"]
        for defn in ("A", "B"):
            for R in res["sum"][defn]:
                for i, j in itertools.combinations(sorted(R), 2):
                    if Js[R][i, j] < 0.9:
                        continue
                    smaller = [R - {k} for k in R if k not in (i, j) and len(R) - 1 >= 2]
                    if not smaller:
                        continue
                    jmin = min(Jt[S][i, j] for S in smaller)
                    if jmin <= 0.5:
                        out["milestone"].append([defn, fmt_set(R), i, j, jmin, Js[R][i, j]])
    out["records"] = recs if (gname == "G1" and v == 0.0 and W == 0.0 and gamma == 0.0) else None
    return out


# ---------------------------------------------------------------- checks
def run_checks(out):
    # gamma -> infinity: records stay |a><a|, every J = 0, no domains
    for gname in ("G1", "G2"):
        V, blocks = graph(gname, 0.2)
        recs = records(V, np.zeros(V.shape[0]), math.inf, 1.0)
        res = analyse(recs, blocks, ["trace", "sum"])
        for R, J in res["trace"]["J"].items():
            for i in range(len(J)):
                for j in range(i + 1, len(J)):
                    if i in R or j in R:
                        L.check(J[i, j] == 0.0, "gamma -> inf: every J involving R is 0 under Phi^trace")
                    else:
                        L.check(abs(J[i, j] - 1) < 1e-12, "two sites outside R both read as 'outside'")
        L.check(not res["trace"]["A"] and not res["trace"]["B"], f"gamma -> inf: no domains under Phi^trace ({gname})")
        lumped = sorted(set(res["sum"]["A"]) | set(res["sum"]["B"]), key=lambda S: (len(S), sorted(S)))
        out.say(f"gamma -> infinity, {gname}: no domains under Phi^trace; under Phi^sum {len(lumped)} sets are still domains, "
                f"e.g. {', '.join(fmt_set(R) for R in lumped[:4])}: the summing channel joins sites in one block by construction.")
    # finite but large gamma approaches the limit
    V, blocks = graph("G1", 0.2)
    recs = records(V, np.zeros(8), 1e5, 1.0)
    Jt = J_quantum(reg_trace(recs, frozenset(range(4))))
    L.check(np.max(Jt[:4] - np.eye(8)[:4]) < 0.05, f"gamma = 1e5: joins involving R near 0 ({np.max(Jt[:4] - np.eye(8)[:4]):.3f})")
    # G1, v = 0, gamma = 0: both clusters are domains (Phi^sum), and J across is exactly 0
    V, blocks = graph("G1", 0.0)
    for t in (1.0, 3.0):
        recs = records(V, np.zeros(8), 0.0, t)
        res = analyse(recs, blocks, ["trace", "sum"])
        A_, B_ = frozenset(range(4)), frozenset(range(4, 8))
        for ch in ("trace", "sum"):
            for R in (A_, B_):
                J = res[ch]["J"][R]
                across = [J[i, j] for i in range(8) for j in range(8) if (i in R) != (j in R)]
                L.check(max(across) < 1e-6, f"G1 v = 0: J across exactly 0 ({ch}): {max(across)}")
        for R in (A_, B_):
            L.check(R in res["sum"]["A"] and R in res["sum"]["B"], "G1 v = 0, gamma = 0: clusters are domains under Phi^sum (A and B)")
        trace_A = all(R in res["trace"]["A"] for R in (A_, B_))
        trace_B = all(R in res["trace"]["B"] for R in (A_, B_))
        sig_trace = res["trace"]["sigma"][A_]
        nb = max(res["trace"]["sigma"][A_ | {k}] for k in range(4, 8))
    out.say(f"G1, v = 0, gamma = 0: both clusters are domains under Phi^sum (A and B), J across = 0 exactly. "
            f"Under Phi^trace sigma(cluster) = {sig_trace:.2e}: unitary evolution keeps records from different sites "
            f"orthogonal, so nothing joins inside. Definition A still {'finds' if trace_A else 'does not find'} the clusters, "
            f"as local maxima at sigma = 0 (adding a site gives sigma up to {nb:.3f}); definition B "
            f"{'finds' if trace_B else 'does not find'} them.")
    # mirror symmetry of G1 at W = 0
    mirror = lambda R: frozenset(7 - i for i in R)
    for gamma in (0.1, 1.0):
        V, blocks = graph("G1", 0.2)
        res = analyse(records(V, np.zeros(8), gamma, 3.0), blocks, ["trace", "sum"])
        for ch in ("trace", "sum"):
            for d in ("A", "B"):
                s = set(res[ch][d])
                L.check(s == {mirror(R) for R in s}, f"G1 mirror symmetry ({ch}, {d})")
    out.say("Checks passed: gamma -> infinity gives J = 0 and no domains; G1 mirror symmetry; monotonicity "
            "(smaller R' joins more under Phi^trace; Phi^sum joins at least as much as Phi^trace); trace preserved.")


def main(quick=False):
    out = L.Output("toy4", "Toy 4: domains on a small graph")
    out.flag("the gamma -> infinity check", "applied to Phi^trace only (J = 0 for every pair with a member in R; no domains)",
             "under Phi^sum, sites in one block read the same by construction, so block subsets stay domains even with no "
             "transport; two sites both outside R also read the same ('outside') under either channel")
    out.flag("numerical floor of the root fidelity", "quantum J below 1e-6 is set to 0; zero is checked to 1e-6, not 1e-12",
             "square roots of eigenvalues at rounding level (1e-16) give about 1e-8, which otherwise reads as structure "
             "(it made the clusters look like domains under Phi^trace at gamma = 0)")
    out.flag("site disorder draws", f"one fixed draw per (N, W): seed {DISORDER_SEED} + N; the same draw for every gamma and t",
             "the spec gives W but not the number of disorder draws")
    out.flag("the G1 check under Phi^trace", "applied to Phi^sum only",
             "under Phi^trace with gamma = 0, unitary evolution keeps records from different sites orthogonal (J = 0 inside "
             "and across), so sigma = 0 and the clusters cannot be domains; the spec's check holds under Phi^sum")
    out.flag("milestone smaller sets", "R' = R minus one site (not i or j); by monotonicity these give the lowest J among smaller sets",
             "searching every smaller set is redundant")
    out.flag("ties in definition A", "a neighbour with equal sigma disqualifies R (strict local maximum)", "the spec says 'lowers'")
    out.flag("G3 frustrated bond", "bond 0-1 set to -1 (triangle 0-1-2 in cluster A)", "the spec names a single negative bond, not which")
    out.flag("G2 Phi^sum blocks", "{0,1}, {2,3}, {4,5}, {6,7}", "the spec says 'pairs of neighbours'")
    run_checks(out)

    jobs = []
    if quick:
        jobs = [("G1", 0.2, 0, 0.0, 1.0, 3.0, ["trace", "sum"]), ("G2", 0.0, 0, 0.0, 1.0, 3.0, ["trace", "sum"])]
    else:
        for gname in ("G1", "G3"):
            for v in V_LIST:
                for W in W_LIST:
                    for g in G_LIST:
                        for t in T_LIST:
                            jobs.append((gname, v, 0, W, g, t, ["trace", "sum"]))
        for W in W_LIST:
            for g in G_LIST:
                for t in T_LIST:
                    jobs.append(("G2", 0.0, 0, W, g, t, ["trace", "sum"]))
                    for s in G4_SEEDS:
                        jobs.append(("G4", 0.0, s, W, g, t, ["trace"]))
    t0 = time.time()
    with Pool(workers()) as pool:
        results = pool.map(point_job, jobs, chunksize=4)
    out.say(f"{len(jobs)} parameter points analysed in {time.time() - t0:.0f} s.")

    cols = ["graph", "v", "G4 seed", "W", "gamma", "t"]
    dom_rows, nest_rows, str_rows, mile_rows, dis_rows = [], [], [], [], []
    for r in results:
        key = list(r["key"])
        for d in r["domains"]:
            dom_rows.append(key + d)
        for (ch, defn), s in r["strongest"].items():
            str_rows.append(key + [ch, defn, s])
        for m in r["milestone"]:
            mile_rows.append(key + m)
        for d in r["disagree"]:
            dis_rows.append(key + d)
        by = {}
        for ch, defn, Rs, size, sig in r["domains"]:
            by.setdefault((ch, defn), []).append(frozenset(int(x) for x in Rs.strip("{}").split(",")))
        for (ch, defn), lst in by.items():
            counts = {"inside": 0, "overlap": 0, "disjoint": 0}
            for R1, R2 in itertools.combinations(lst, 2):
                rel = relation(R1, R2)
                counts["inside" if rel in ("inside", "contains") else rel] += 1
            nest_rows.append(key + [ch, defn, len(lst), counts["inside"], counts["overlap"], counts["disjoint"]])
    hdr = {**HEADER, "channels": "Phi^trace (block on R + outside); Phi^sum (block populations + outside)",
           "disorder": f"one draw per W, seed {DISORDER_SEED} + N", "G4 seeds": "0..9"}
    out.table("1_domains", hdr, cols + ["channel", "definition", "R", "size", "sigma"], dom_rows,
              note="Sorted by sigma within each point. The full set R = all sites has no 'across' pairs and is excluded; "
                   "its edge with the environment is outside this toy.")
    out.table("2_nesting", hdr, cols + ["channel", "definition", "domains", "nested pairs", "overlapping pairs", "disjoint pairs"], nest_rows)
    out.table("3_strongest_sigma", hdr, cols + ["channel", "definition", "sigma of strongest domain"], str_rows)
    out.table("4_milestone", hdr, cols + ["definition", "R (domain under Phi^sum)", "i", "j", "min J under Phi^trace of R minus one", "J^R under Phi^sum"],
              mile_rows, note="Pairs cut at a finer grain (J <= 0.5 under the trace channel of a smaller set) and joined at R's grain "
                              "(J >= 0.9 under Phi^sum).")
    out.table("5_A_vs_B", hdr, cols + ["channel", "R", "found by", "sigma"], dis_rows)

    # summary numbers
    n_dom = len(dom_rows)
    out.say(f"Domains found (all points, channels, definitions): {n_dom}; milestone pairs: {len(mile_rows)}; A/B disagreements: {len(dis_rows)}.")
    for ch in ("trace", "sum"):
        for gname in ("G1", "G3", "G2", "G4"):
            sel = [x for x in str_rows if x[0] == gname and x[6] == ch and x[7] == "A" and x[2] == 0 and x[3] == 0.0]
            if not sel:
                continue
            best = {}
            for x in sel:
                best.setdefault(x[4], []).append(x[8])
            line = ", ".join(f"gamma {g:g}: {np.nanmax(v) if not all(np.isnan(v)) else float('nan'):.3f}" for g, v in sorted(best.items()))
            out.say(f"{gname}, Phi^{ch}, definition A, W = 0 (max over v, t): strongest sigma by gamma -> {line}")
    out.finish()
    return results


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
