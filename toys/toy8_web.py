"""Toy 8: the monitored web (calibration).

A ring of L qubits; a reference qubit starts in a Bell pair with ring qubit 0. The question q is the
reference's state, 0 or 1 (the thread). Conditional on q = a, ring qubit 0 starts in |a> (the others in
|0>), so the toy simulates the two alternatives as two ring branches with the same gates and the same
record times. Each layer applies Haar-random two-qubit gates on alternating bonds (brickwork); then
each qubit is recorded in Z with probability p, and the outcome is kept as a classical record bit.
Feedback variant: after outcome 1, with probability f, the qubit is flipped back to 0.

Records are sampled from the average of the two alternatives' record distributions, p_bar = (p0 + p1)/2,
which is also the Born distribution of the real Bell-pair experiment. Each trajectory carries p0(m),
p1(m) and both branch states given m.

Quantities (records classical unless stated):
  J over M          sum_m sqrt(p0 p1) (estimator: mean of sqrt(p0 p1)/p_bar)
  J over S and M    sum_m sqrt(p0 p1) |<psi0_m|psi1_m>|   (classical records)
  J global          |sum_m sqrt(p0 p1) <psi0_m|psi1_m>|   records kept coherent: exactly 0 (checked by enumeration)
  subset curve      J over M plus k ring qubits, exact minimum over all C(L, k) sets, k = 0..3
  S_ref             entropy of the reference given the records (the purification order parameter)
  S_half            entropy of half the ring given the records; tripartite informations I3(A:B:C) over ring
                    quarters and I3(R:A:B) with the reference and ring halves
  R_delta           single record bits each holding >= (1 - delta) of I(q; M), marginals by importance weights
J over S alone (records discarded) needs an average over exponentially many record strings and is not
computed (adopted 17 Nov); S_ref and the ring overlap given the records stand in for it.

Run: python toys/toy8_web.py            defaults: L = 8, 12, 16; T = 2L (snapshot at T = L); f = 0, 1; 100 realisations
     python toys/toy8_web.py --L20      adds L = 20 (slow)
     python toys/toy8_web.py --quick    checks and a small sweep
"""

import itertools
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge_lib as L  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas.stamp import workers  # noqa: E402

P_GRID = sorted(set([round(x, 2) for x in np.arange(0, 0.501, 0.02)] + [round(x, 2) for x in np.arange(0.10, 0.251, 0.01)]))
P_SUBSET = [0.05, 0.17, 0.4]
F_LIST = [0.0, 1.0]
N_REAL = 100
PC_PUBLISHED = 0.17


# ---------------------------------------------------------------- statevector helpers (axis 0 = branch)
def haar4(rng):
    Z = (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))) / math.sqrt(2)
    Q, R = np.linalg.qr(Z)
    return (Q * (np.diag(R) / np.abs(np.diag(R)))).reshape(2, 2, 2, 2)


def apply2(psi, U, i, j):
    out = np.tensordot(U, psi, axes=([2, 3], [i + 1, j + 1]))
    return np.moveaxis(out, [0, 1], [i + 1, j + 1])


def apply_x(psi, i):
    return np.flip(psi, axis=i + 1)


def prob1(psi, i):
    sl = [slice(None)] * psi.ndim
    sl[i + 1] = 1
    sub = psi[tuple(sl)]
    return np.sum(np.abs(sub.reshape(2, -1)) ** 2, axis=1)


def project(psi, i, o):
    sl = [slice(None)] * psi.ndim
    sl[i + 1] = 1 - o
    psi = psi.copy()
    psi[tuple(sl)] = 0
    return psi


def initial(L_):
    psi = np.zeros((2,) + (2,) * L_, complex)
    psi[(0,) + (0,) * L_] = 1.0
    psi[(1, 1) + (0,) * (L_ - 1)] = 1.0
    return psi


def entropy_of(vec, n, X):
    X = list(X)
    rest = [k for k in range(n) if k not in X]
    M = np.transpose(vec.reshape([2] * n), X + rest).reshape(2 ** len(X), -1)
    s = np.linalg.svd(M, compute_uv=False) ** 2
    s = s[s > 1e-14]
    return float(-np.sum(s * np.log2(s)))


def reduced_k(psi_a, K, L_):
    others = [k for k in range(L_) if k not in K]
    t = np.tensordot(psi_a, psi_a.conj(), axes=(others, others))
    d = 2 ** len(K)
    return t.reshape(d, d)


# ---------------------------------------------------------------- one trajectory
def trajectory(L_, p, f, T, seed, snapshots, want_subsets=False, forced=None):
    """Simulate both branches. forced: optional (placements, outcomes) for exact enumeration."""
    rng = np.random.default_rng(seed)
    psi = initial(L_)
    logw = np.zeros(2)
    alive = np.array([True, True])
    record = -np.ones((T, L_), int)
    snaps = {}
    k_meas = 0
    for layer in range(T):
        start = layer % 2
        for i in range(start, L_ - (0 if L_ % 2 == 0 else 1), 2):
            j = (i + 1) % L_
            if i == j:
                continue
            psi = apply2(psi, haar4(rng), i, j)
        for i in range(L_):
            if forced is not None:
                measured = forced[0][layer, i]
            else:
                measured = rng.random() < p
            if not measured:
                continue
            p1 = prob1(psi, i)
            if forced is not None:
                o = forced[1][k_meas]; k_meas += 1
            else:
                w = np.exp(logw - logw.max()) * alive
                P1 = float(np.sum(w * p1) / np.sum(w))
                o = int(rng.random() < P1)
            po = p1 if o == 1 else 1 - p1
            for a in (0, 1):
                if po[a] <= 1e-300:
                    alive[a] = False
                else:
                    logw[a] += math.log(po[a])
            psi = project(psi, i, o)
            nrm = np.sqrt(np.sum(np.abs(psi.reshape(2, -1)) ** 2, axis=1))
            psi = psi / np.where(nrm > 0, nrm, 1).reshape((2,) + (1,) * L_)
            record[layer, i] = o
            fb = (forced is None and rng.random() < f) or (forced is not None and f >= 1)
            if o == 1 and fb:
                psi = apply_x(psi, i)
        if layer + 1 in snapshots:
            snaps[layer + 1] = measure_snapshot(psi, logw, alive, L_, want_subsets and layer + 1 == max(snapshots))
    return snaps, record, logw, alive, psi


def measure_snapshot(psi, logw, alive, L_, want_subsets):
    p0 = math.exp(logw[0]) if alive[0] else 0.0
    p1 = math.exp(logw[1]) if alive[1] else 0.0
    pbar = (p0 + p1) / 2
    v0, v1 = psi[0].reshape(-1), psi[1].reshape(-1)
    ov = complex(np.vdot(v0, v1)) if (alive[0] and alive[1]) else 0.0
    wM = math.sqrt(p0 * p1) / pbar if pbar > 0 else 0.0         # weight for the estimator of sum_m sqrt(p0 p1) X(m)
    # reference given records: rho = [[p0, sqrt(p0 p1) <psi1|psi0>], [., p1]] / (p0 + p1)
    a, b = p0 / (p0 + p1), p1 / (p0 + p1)
    c = math.sqrt(a * b) * abs(ov)
    lam = 0.5 + math.sqrt(max(0.0, 0.25 - (a * b - c * c)))
    S_ref = L.h2(min(1.0, max(0.0, lam)))
    # the physical state of reference + ring given the records
    phys = np.concatenate([math.sqrt(a) * v0, math.sqrt(b) * v1]) if (alive[0] and alive[1]) else (v0 if alive[0] else v1)
    n = L_ + 1 if (alive[0] and alive[1]) else L_
    off = 1 if n == L_ + 1 else 0
    half = [off + k for k in range(L_ // 2)]
    S_half = entropy_of(phys, n, half)
    q = L_ // 4
    A, B, C = ([off + k for k in range(s, s + q)] for s in (0, q, 2 * q))
    S = lambda X: entropy_of(phys, n, X)
    I3_ring = S(A) + S(B) + S(C) - S(A + B) - S(A + C) - S(B + C) + S(A + B + C)
    if off:
        Rr = [0]; Ah = [1 + k for k in range(L_ // 2)]; Bh = [1 + k for k in range(L_ // 2, L_)]
        I3_ref = S(Rr) + S(Ah) + S(Bh) - S(Rr + Ah) - S(Rr + Bh) - S(Ah + Bh)      # S(RAB) = 0, global pure
    else:
        I3_ref = 0.0
    snap = {"wM": wM, "wSM": wM * abs(ov), "glob": (math.sqrt(p0 * p1) / pbar) * ov if pbar > 0 else 0.0,
            "S_ref": S_ref, "S_half": S_half, "I3_ring": I3_ring, "I3_ref": I3_ref, "ov": abs(ov),
            "w0": p0 / pbar if pbar > 0 else 0.0, "w1": p1 / pbar if pbar > 0 else 0.0}
    if want_subsets:
        sub = {0: wM}
        for k in (1, 2, 3):
            best = None
            for K in itertools.combinations(range(L_), k):
                if not (alive[0] and alive[1]):
                    F = 0.0
                else:
                    F = float(L.fidelity(reduced_k(psi[0], list(K), L_), reduced_k(psi[1], list(K), L_)))
                best = (F, K) if best is None or F < best[0] else best
            sub[k] = best
        snap["subsets"] = sub
    return snap


def job(args):
    L_, p, f, T, seeds, snapshots, want_subsets = args
    res = []
    for s in seeds:
        snaps, record, logw, alive, _ = trajectory(L_, p, f, T, s, snapshots, want_subsets)
        last = snaps[max(snapshots)]
        res.append((s, snaps, record if want_subsets else None, last["w0"], last["w1"]))
    return (L_, p, f), res


# ---------------------------------------------------------------- checks
def enumeration_check(L_=4, T=3, p=0.5, f=0.0, seed=11):
    """Exact sum over every record string for one fixed placement of records: the global joint
    overlap sum_m sqrt(p0 p1) <psi0_m|psi1_m> must be 0 (to 1e-10), and sum_m p_a(m) = 1."""
    rng = np.random.default_rng(seed)
    placements = rng.random((T, L_)) < p
    n_meas = int(placements.sum())
    total, norm0, norm1 = 0j, 0.0, 0.0
    for outcomes in itertools.product((0, 1), repeat=n_meas):
        snaps, record, logw, alive, psi = trajectory(L_, p, f, T, seed + 1, [T], forced=(placements, list(outcomes)))
        p0 = math.exp(logw[0]) if alive[0] else 0.0
        p1 = math.exp(logw[1]) if alive[1] else 0.0
        norm0 += p0; norm1 += p1
        if alive[0] and alive[1]:
            total += math.sqrt(p0 * p1) * np.vdot(psi[0].reshape(-1), psi[1].reshape(-1))
    L.check(abs(norm0 - 1) < 1e-10 and abs(norm1 - 1) < 1e-10, f"record probabilities sum to 1: {norm0}, {norm1}")
    L.check(abs(total) < 1e-10, f"global joint J = |sum_m sqrt(p0 p1) <psi0|psi1>| = {abs(total)} (must be 0)")
    return n_meas, abs(total)


def run_checks(out):
    for f in (0.0, 1.0):
        n, tot = enumeration_check(f=f)
        out.say(f"Exact enumeration (L = 4, 3 layers, {n} records, f = {f:g}): global joint J = {tot:.1e}; "
                "every record string's probabilities sum to 1. No event raises the join, by unitarity.")
    # p = 0: no records, J over M = 1
    snaps, *_ = trajectory(6, 0.0, 0.0, 6, 3, [6])
    L.check(abs(snaps[6]["wM"] - 1) < 1e-12, "p = 0: J over M = 1")
    # p = 1: after one layer the ring is fully recorded; given the records the ring states are identical
    snaps, *_ = trajectory(6, 1.0, 0.0, 1, 5, [1])
    L.check(abs(snaps[1]["ov"] - 1) < 1e-10, "p = 1: after one layer, the ring given the records is the same for both alternatives")
    vals = [trajectory(6, 1.0, 0.0, 1, s, [1])[0][1]["wM"] for s in range(200)]
    out.say(f"p = 1, after one layer: the ring given the records holds nothing of q (overlap 1); J over M = {np.mean(vals):.3f} "
            "(mean of 200 trajectories), not 0: the layer's gate mixes before the first record, so the records hold the thread "
            "only partly and, with classical records, the rest is held by neither (spec check revised, flagged).")


# ---------------------------------------------------------------- main
def main(quick=False, with_L20=False):
    out = L.Output("toy8", "Toy 8: the monitored web (calibration)")
    out.flag("ring initial state", "ring qubit 0 in |q>, all others |0>", "the spec fixes the Bell pair, not the other qubits")
    out.flag("brickwork on a ring", "even layers bonds (0,1),(2,3)...; odd layers (1,2),...,(L-1,0)", "periodic boundary, as 'ring'")
    out.flag("sizes and sweep (adopted 17 Nov)", "L = 8, 12, 16 (L = 20 with --L20); T = 2L with a snapshot at T = L; f = 0 and 1; "
             "100 realisations per point; p on the spec grid", "the full spec would take days on a laptop")
    out.flag("J over S alone", "not computed; the reference's entropy given the records and the ring overlap given the records are "
             "reported instead", "records discarded means averaging over exponentially many record strings (adopted 17 Nov)")
    out.flag("classical vs coherent records", "J over S and M is reported with classical records (sum of |overlaps|); the global "
             "joint J with coherent records (|sum of overlaps|) is 0 by unitarity and checked exactly by enumeration",
             "with classical records the joint J can exceed 0: the spec's 'joint stays 0' holds only for coherent records")
    out.flag("p = 1 check", "replaced: after one layer the ring given the records holds nothing of q (checked); J over M after one "
             "layer is reported, not asserted to be 0", "the gate acts before the first record, so the records hold the thread only partly")
    out.flag("subset curves (adopted 17 Nov)", "J over M plus the k ring qubits that minimise it, exact over all C(L, k) sets for "
             "k = 0..3, at p = 0.05, 0.17, 0.4 and every L; k_half reported if reached by k = 3", "record bits enter as one block")
    out.flag("tripartite informations", "I3(A:B:C) over the first three ring quarters (Zabalo et al.) and I3(R:A:B) with the "
             "reference and the two ring halves", "the spec's 'three ring sections with the reference' admits both readings")
    out.flag("R_delta", "single record bits each holding >= (1 - delta) of I(q; M); per-bit marginals from the same trajectories "
             "by importance weights p_a/p_bar; 'not recorded' counted as a third symbol", "a lower bound on Zurek's R_delta")
    out.flag("p_c check", "growth of S_half from L = 8 to 16 must exceed 0.5 bits at p = 0.05 and be below 0.3 bits at p = 0.4; "
             "the crossing is reported, not asserted to 0.17 +/- 0.03", "L <= 16 is too small for the spec's +/- 0.03")
    run_checks(out)

    Ls = [4, 6] if quick else [8, 12, 16] + ([20] if with_L20 else [])
    ps = [0.0, 0.1, 0.2, 0.4] if quick else P_GRID
    fs = [0.0, 1.0]
    nreal = 8 if quick else N_REAL
    jobs = []
    for L_ in Ls:
        T = 2 * L_
        for ip, p in enumerate(ps):
            for fi, f in enumerate(fs):
                want = (p in P_SUBSET) or quick
                seeds = [100000 * L_ + 1000 * ip + 100 * fi + r for r in range(nreal)]
                chunk = 10 if L_ >= 12 else nreal
                for c in range(0, nreal, chunk):
                    jobs.append((L_, p, f, T, seeds[c:c + chunk], [L_, T], want))
    t0 = time.time()
    with Pool(workers()) as pool:
        results = pool.map(job, jobs, chunksize=1)
    out.say(f"{sum(len(r) for _, r in results)} trajectories in {time.time() - t0:.0f} s.")
    agg = {}
    for key, res in results:
        agg.setdefault(key, []).extend(res)

    rows1, rows3, rows2, rows4 = [], [], [], []
    S_half_at = {}
    for (L_, p, f), res in sorted(agg.items()):
        for T in (L_, 2 * L_):
            S = [r[1][T] for r in res]
            m = lambda k: float(np.mean([s[k] for s in S]))
            se = lambda k: float(np.std([s[k] for s in S]) / math.sqrt(len(S)))
            glob = abs(np.mean([s["glob"] for s in S]))
            rows1.append([L_, p, f, T, m("wM"), se("wM"), m("wSM"), glob, m("ov")])
            rows3.append([L_, p, f, T, m("S_ref"), m("S_half"), se("S_half"), m("I3_ring"), m("I3_ref")])
            if T == 2 * L_ and f == 0:
                S_half_at[(L_, p)] = m("S_half")
        last = [r[1][2 * L_] for r in res]
        if "subsets" in last[0]:
            seeds = [r[0] for r in res]
            curve = [float(np.mean([s["subsets"][0] for s in last]))]
            for k in (1, 2, 3):
                curve.append(float(np.mean([s["wM"] * s["subsets"][k][0] for s in last])))
            kh = next((k for k, v in enumerate(curve) if v <= 0.5), None)
            rows2.append([L_, p, f, 2 * L_] + curve + [kh if kh is not None else "> 3", (kh / L_) if kh else float("nan")])
            # R_delta from importance-weighted single-bit marginals
            recs = [r[2] for r in res]
            w0 = np.array([r[3] for r in res]); w1 = np.array([r[4] for r in res])
            IqM = 1 - float(np.mean([L.h2(a / (a + b)) if a + b > 0 else 0 for a, b in zip(w0, w1)]))
            Tn, Ln = recs[0].shape
            count = 0
            for t in range(Tn):
                for i in range(Ln):
                    sym = np.array([rec[t, i] for rec in recs])
                    P0 = np.array([np.mean(w0 * (sym == x)) for x in (-1, 0, 1)])
                    P1 = np.array([np.mean(w1 * (sym == x)) for x in (-1, 0, 1)])
                    P0 = P0 / P0.sum() if P0.sum() > 0 else P0; P1 = P1 / P1.sum() if P1.sum() > 0 else P1
                    I_bit = L.mutual_information([0.5, 0.5], [P0, P1])
                    if IqM > 1e-6 and I_bit >= (1 - L.DELTA_R) * IqM:
                        count += 1
            rows4.append([L_, p, f, IqM, count])
    hdr = {"event": "the late event on ring qubit 0 at time T", "question": "the reference's state, 0 or 1 (the thread)",
           "prior": "uniform", "records": "Z outcomes, kept as classical bits",
           "seeds": "100000 L + 1000 i_p + 100 i_f + r"}
    out.table("1_where_the_thread_lives", hdr, ["L", "p", "f", "T", "J over M", "se", "J over S and M (classical records)",
                                                "global joint J (coherent records)", "ring overlap given records"], rows1,
              note="Global joint J is an estimate of a quantity that is exactly 0 (see the enumeration check); its size here is sampling noise.")
    out.table("2_subset_curves", hdr, ["L", "p", "f", "T", "J (records only, k = 0)", "k = 1", "k = 2", "k = 3", "k_half", "k_half / L"], rows2)
    out.table("3_entropies", hdr, ["L", "p", "f", "T", "S_ref given records", "S_half given records", "se", "I3(A:B:C) ring quarters", "I3(R:A:B)"], rows3)
    out.table("4_redundancy", hdr, ["L", "p", "f", "I(q; M) bits", "R_delta (single record bits)"], rows4)

    if not quick:
        lo, hi = min(Ls), max(Ls)
        growth = {p: S_half_at[(hi, p)] - S_half_at[(lo, p)] for p in ps if (hi, p) in S_half_at and (lo, p) in S_half_at}
        L.check(growth[0.06 if 0.06 in growth else min(growth, key=lambda x: abs(x - 0.05))] > 0.5, "S_half grows with L at small p")
        L.check(growth[0.4] < 0.3, "S_half does not grow with L at p = 0.4")
        cross = next((p for p in sorted(growth) if growth[p] < 0.5 * growth[min(growth)]), float("nan"))
        out.say(f"Half-ring entropy growth from L = {lo} to {hi}: {growth[min(growth, key=lambda x: abs(x - 0.05))]:.2f} bits near p = 0.05, "
                f"{growth[0.4]:.2f} bits at p = 0.4; growth halves by p = {cross} (published p_c = {PC_PUBLISHED} for Haar qubits).")
    for r in rows2:
        out.say(f"Subset curve, L = {r[0]}, p = {r[1]}, f = {r[2]:g}: J = {r[4]:.3f}, {r[5]:.3f}, {r[6]:.3f}, {r[7]:.3f} for k = 0..3; k_half = {r[8]}")
    out.say(L.SATURATION_NOTE)
    out.finish()


if __name__ == "__main__":
    main(quick="--quick" in sys.argv, with_L20="--L20" in sys.argv)
