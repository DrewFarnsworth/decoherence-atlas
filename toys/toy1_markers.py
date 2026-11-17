"""Toy 1: one system, many partial markers (spec: Bridge toys specification).

System S: one qubit in (|0> + |1>)/sqrt2; alternatives a in {0, 1}, uniform prior.
N marker qubits; if S = a, marker k ends in |m_k^a>, with <m_k^0|m_k^1> = c_k (real).
Records rho_a = |M^a><M^a|, the product of the markers.

Pure records have closed forms, and the code checks them to 1e-10 against explicit
state vectors and an explicit partial trace for small N. Noisy reads of k identical
markers are computed exactly for any k with the Schur-Weyl decomposition: a
permutation-invariant operator X^{(x)k} acts on the spin-j block as
det(X)^{k/2-j} Sym^{2j}(X), with multiplicity C(k, k/2-j) - C(k, k/2-j-1).

Run: python toys/toy1_markers.py
"""

import math
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge_lib as L  # noqa: E402

N_LIST = [1, 2, 5, 10, 20, 50, 100]
C_LIST = [0.999, 0.99, 0.95, 0.9, 0.7, 0.42, 0.0]
Q_LIST = [0.0, 0.5, 0.9]
DELTA = 0.1
SEEDS = list(range(20))
K_NOISY_MAX = 12

HEADER = {
    "event": "the system S takes path 0 or path 1",
    "alternatives": "a in {0, 1}",
    "prior": "uniform (1/2, 1/2)",
}


def hb(x):
    return L.h2((1 + x) / 2)


# ---------------------------------------------------------------- closed forms (real overlaps)
def info_fragment(C_all, C_F, C_rest):
    """I(S:F) for the pure global state, from overlaps of all markers, the fragment and the rest."""
    return hb(C_all) + hb(C_F) - hb(C_rest)


# ---------------------------------------------------------------- explicit checks for small N
def marker(c):
    return np.array([1.0, 0.0]), np.array([c, math.sqrt(max(0.0, 1 - c * c))])


def global_state(cs):
    """(|0>|M0> + |1>|M1>)/sqrt2 as a vector over S (x) markers."""
    M0, M1 = np.array([1.0]), np.array([1.0])
    for c in cs:
        m0, m1 = marker(c)
        M0, M1 = np.kron(M0, m0), np.kron(M1, m1)
    return (np.kron([1.0, 0.0], M0) + np.kron([0.0, 1.0], M1)) / math.sqrt(2), M0, M1


def explicit_check(cs, k):
    """Compare closed forms with explicit vectors and partial traces (small N only)."""
    N = len(cs)
    psi, M0, M1 = global_state(cs)
    rho = np.outer(psi, psi)
    dims = [2] * (N + 1)
    F_idx = list(range(1, k + 1))
    # fragment records
    if k > 0:
        r0 = L.partial_trace(np.outer(M0, M0), [2] * N, list(range(k)))
        r1 = L.partial_trace(np.outer(M1, M1), [2] * N, list(range(k)))
        J = float(L.fidelity(r0, r1)); D = float(L.trace_distance(r0, r1))
    else:
        J, D = 1.0, 0.0
    CF = float(np.prod(cs[:k])) if k else 1.0
    L.check(abs(J - CF) < 1e-10, f"J^k closed form: {J} vs {CF}")
    L.check(abs(D - math.sqrt(1 - CF * CF)) < 1e-10, "D^k closed form")
    # quantum mutual information S:F from the explicit global state
    S_S = L.vn_entropy(L.partial_trace(rho, dims, [0]))
    S_F = L.vn_entropy(L.partial_trace(rho, dims, F_idx)) if k else 0.0
    S_SF = L.vn_entropy(L.partial_trace(rho, dims, [0] + F_idx))
    I_exp = S_S + S_F - S_SF
    C_all = float(np.prod(cs)); C_rest = float(np.prod(cs[k:])) if k < N else 1.0
    L.check(abs(I_exp - info_fragment(C_all, CF, C_rest)) < 1e-10, "I(S:F_k) closed form")
    chi = L.holevo([0.5, 0.5], [r0, r1]) if k else 0.0
    L.check(abs(chi - hb(CF)) < 1e-10, "chi_k closed form")


# ---------------------------------------------------------------- exact noisy reads (Schur-Weyl)
def sym_power(X, n):
    """Sym^n(X) for a 2x2 matrix X, in the orthonormal basis of the symmetric subspace."""
    if n == 0:
        return np.array([[1.0 + 0j]])
    a, b = X[0, 0], X[0, 1]
    c, d = X[1, 0], X[1, 1]
    # X acts on e0 -> a e0 + c e1, e1 -> b e0 + d e1; monomial e0^(n-i) e1^i
    S = np.zeros((n + 1, n + 1), dtype=complex)
    for i in range(n + 1):
        # (a e0 + c e1)^(n-i) (b e0 + d e1)^i expanded
        for p in range(n - i + 1):          # p factors of c e1 from the first bracket
            for r in range(i + 1):          # r factors of d e1 from the second
                coef = comb(n - i, p) * comb(i, r) * a ** (n - i - p) * c ** p * b ** (i - r) * d ** r
                S[p + r, i] += coef
    norm = np.array([math.sqrt(comb(n, i)) for i in range(n + 1)])
    # symmetrizing maps the tensor monomial m_i to C(n, i) x^(n-i) y^i, so
    # <p| X^(x)n |i> = S[p, i] sqrt(C(n, i)) / sqrt(C(n, p))
    return S * norm[None, :] / norm[:, None]


def blocks(k):
    """(j2 = 2j, multiplicity) for k qubits."""
    out = []
    for s in range(0, k // 2 + 1):            # s = k/2 - j
        mult = comb(k, s) - (comb(k, s - 1) if s > 0 else 0)
        if mult > 0:
            out.append((k - 2 * s, s, mult))
    return out


def tensor_power_spectrum(X, Y, k, wx=1.0, wy=1.0):
    """Eigenvalues (with multiplicities) of wx X^(x)k + wy Y^(x)k for Hermitian 2x2 X, Y."""
    vals, mults = [], []
    dX, dY = np.linalg.det(X), np.linalg.det(Y)
    for n, s, mult in blocks(k):
        B = wx * dX ** s * sym_power(X, n) + wy * dY ** s * sym_power(Y, n)
        w = np.linalg.eigvalsh((B + B.conj().T) / 2)
        vals.append(w); mults.append(np.full(len(w), mult))
    return np.concatenate(vals), np.concatenate(mults)


def noisy_read(c, q, k):
    """J, D and chi for a fragment of k identical markers read through depolarizing noise q."""
    m0, m1 = marker(c)
    r0, r1 = L.depolarize(L.dm(m0), q), L.depolarize(L.dm(m1), q)
    Fq = float(L.fidelity(r0, r1))
    J = Fq ** k
    if k == 0:
        return 1.0, 0.0, 0.0, Fq
    w, m = tensor_power_spectrum(r0, r1, k, 1.0, -1.0)
    D = 0.5 * float(np.sum(m * np.abs(w)))
    wa, ma = tensor_power_spectrum(r0, r1, k, 0.5, 0.5)
    wa = np.clip(wa, 0, None)
    S_avg = float(-np.sum(ma * np.where(wa > 1e-300, wa * np.log2(np.where(wa > 0, wa, 1)), 0)))
    chi = S_avg - k * L.vn_entropy(r0)          # both records have the same entropy
    return J, D, chi, Fq


def noisy_dense(c, q, k):
    m0, m1 = marker(c)
    r0, r1 = L.depolarize(L.dm(m0), q), L.depolarize(L.dm(m1), q)
    R0, R1 = np.array([[1.0]]), np.array([[1.0]])
    for _ in range(k):
        R0, R1 = np.kron(R0, r0), np.kron(R1, r1)
    return float(L.fidelity(R0, R1)), float(L.trace_distance(R0, R1)), L.holevo([0.5, 0.5], [R0, R1])


# ---------------------------------------------------------------- main
def main():
    out = L.Output("toy1", "Toy 1: one system, many partial markers")
    out.flag("N_cut over all integers", "N_cut(c) = smallest integer N with c^N <= 0.1, not only N in the sweep",
             "the sweep list is too coarse to locate the crossing")
    out.flag("noisy reads method", "exact Schur-Weyl block decomposition for identical markers; dense check for k <= 6",
             "dense 2^12 matrices are slow; the block method is exact and is verified against dense")
    out.flag("heterogeneous variant: which k", "fragments at k = round(f N) for f in 0.1..1.0, and every k for N <= 20",
             "the spec fixes how many fragments per k, not which k values to sample")

    # explicit checks for small N (uniform and heterogeneous c)
    rng = np.random.default_rng(0)
    for N in (1, 2, 5, 9):
        for c in (0.95, 0.42, 0.0):
            for k in range(N + 1):
                explicit_check([c] * N, k)
        cs = list(rng.uniform(0.5, 1.0, N))
        for k in range(N + 1):
            explicit_check(cs, k)
    out.say("Closed forms checked against explicit states and partial traces (N up to 9): all agree to 1e-10.")

    # 1. compounding table
    rows = []
    for c in C_LIST:
        ncut = 1 if c == 0 else math.ceil(math.log(0.1) / math.log(c))
        for N in N_LIST:
            J = c ** N
            D = math.sqrt(1 - J * J)
            L.check_pair(J, D, pure=True, label=f"compounding c={c} N={N}")
            rows.append([c, N, J, D, hb(J), ncut])
    out.table("1_compounding", {**HEADER, "channel": "identity on all N markers (J_phys)", "time": "after marking"},
              ["c", "N", "J_phys", "D", "chi bits", "N_cut(c)"], rows, note=L.SATURATION_NOTE)
    for c in (0.999, 0.99, 0.95, 0.9, 0.7):
        out.say(f"N_cut(c = {c}) = {math.ceil(math.log(0.1) / math.log(c))}")

    # 2. partial-information data and 3. redundancy
    rows, red = [], []
    for c in (0.95, 0.9, 0.7):
        N = 50
        HS = hb(c ** N)
        for k in range(N + 1):
            I = info_fragment(c ** N, c ** k, c ** (N - k))
            I2 = info_fragment(c ** N, c ** (N - k), c ** k)
            L.check(abs(I + I2 - 2 * HS) < 1e-10, "I(S:F_k) + I(S:F_N-k) = 2 H_S")
            if k == 0:
                L.check(abs(I) < 1e-12, "I(S:F_0) = 0")
            if k == N:
                L.check(abs(I - 2 * HS) < 1e-10, "I(S:F_N) = 2 H_S")
            rows.append([c, k / N, c ** k, hb(c ** k), I, HS])
    out.table("2_partial_information", {**HEADER, "N": 50, "channel": "partial trace onto k markers",
                                        "time": "after marking"},
              ["c", "k/N", "J^k", "chi_k bits", "I(S:F_k) bits", "H_S bits"], rows)
    for c in C_LIST:
        for N in N_LIST:
            HS = hb(c ** N)
            kd = next((k for k in range(N + 1) if info_fragment(c ** N, c ** k, c ** (N - k)) >= (1 - DELTA) * HS - 1e-12), None)
            red.append([c, N, HS, kd, (N / kd) if kd else float("nan")])
    out.table("3_redundancy", {**HEADER, "delta": DELTA, "channel": "partial trace onto k markers"},
              ["c", "N", "H_S bits", "k_delta", "R_delta = N/k_delta"], red,
              note="k_delta = 0 cannot occur: a fragment of no markers holds no information. "
                   "Where H_S is near 0 the system is barely marked and R_delta is not meaningful.")
    for c, N, HS, kd, R in red:
        if N == 100 and c in (0.95, 0.9, 0.7):
            out.say(f"N = 100, c = {c}: H_S = {HS:.4f} bits, k_delta = {kd}, R_delta = {R:.1f}")

    # 4. physical against registered
    rows = []
    for c in C_LIST:
        for N in N_LIST:
            Jp = c ** N
            ks = [k for k in range(N + 1) if Jp <= 0.1 and c ** k >= 0.9]
            for k in range(N + 1):
                L.check_monotone(c ** k, Jp, label="J^k >= J_phys")
            rows.append([c, N, Jp, (f"{max(1, min(ks))}-{max(ks)}" if ks and max(ks) >= 1 else "none"),
                         (max(ks) / N if ks else float("nan"))])
    out.table("4_physical_vs_registered", {**HEADER, "channel": "J_phys: all markers; J^k: k markers"},
              ["c", "N", "J_phys", "k range: J_phys <= 0.1 and J^k >= 0.9", "largest such k / N"], rows)
    for c, N, Jp, rng_k, frac in rows:
        if N == 100 and c in (0.999, 0.99, 0.95):
            if rng_k == "none":
                out.say(f"N = 100, c = {c}: J_phys = {Jp:.3g}; no fragment is both inside a physical cut and still joined")
            else:
                out.say(f"N = 100, c = {c}: J_phys = {Jp:.3g}, physically cut; fragments of {rng_k} markers still find the paths joined (J^k >= 0.9)")

    # 5. noisy reads
    for c in (0.9, 0.42):
        for q in (0.5, 0.9):
            for k in range(1, 7):
                Jd, Dd, chid = noisy_dense(c, q, k)
                J, D, chi, _ = noisy_read(c, q, k)
                L.check(abs(J - Jd) < 1e-9 and abs(D - Dd) < 1e-9 and abs(chi - chid) < 1e-9,
                        f"Schur-Weyl against dense, c={c} q={q} k={k}: {(J, D, chi)} vs {(Jd, Dd, chid)}")
    rows = []
    for c in C_LIST:
        for q in Q_LIST:
            for k in range(0, K_NOISY_MAX + 1):
                J, D, chi, Fq = noisy_read(c, q, k)
                L.check_pair(J, D, pure=(q == 0), tol=1e-9, label=f"noisy c={c} q={q} k={k}")
                L.check_monotone(J, c ** k, label="J^{k,q} >= J^k")
                if q == 0:
                    L.check(abs(chi - hb(c ** k)) < 1e-9, "noisy read at q = 0 matches chi_k")
                rows.append([c, q, k, J, D, chi])
    out.table("5_noisy_reads", {**HEADER, "channel": "partial trace onto k markers, then depolarizing q on each",
                                "time": "after marking"},
              ["c", "q", "k", "J^{k,q}", "D^{k,q}", "chi_{k,q} bits"], rows)
    J, D, chi, _ = noisy_read(0.42, 0.5, 12)
    out.say(f"Noisy read, c = 0.42, q = 0.5, k = 12: J = {J:.4f}, D = {D:.4f}, chi = {chi:.4f} bits")

    # heterogeneous variant
    rows = []
    for cmin in (0.5, 0.9):
        for N in N_LIST:
            Jps, kds = [], []
            for seed in SEEDS:
                r = np.random.default_rng(1000 * N + seed + (0 if cmin == 0.5 else 7))
                cs = r.uniform(cmin, 1.0, N)
                C_all = float(np.prod(cs))
                HS = hb(C_all)
                Jps.append(C_all)
                ks = range(N + 1) if N <= 20 else sorted({max(1, round(f * N)) for f in np.linspace(0.1, 1.0, 10)})
                kd = None
                for k in ks:
                    n_frag = comb(N, k)
                    if n_frag <= 1000:
                        frags = [list(F) for F in __import__("itertools").combinations(range(N), k)]
                    else:
                        frags = [list(r.choice(N, k, replace=False)) for _ in range(200)]
                    Is = []
                    for F in frags:
                        Fs = set(F)
                        CF = float(np.prod([cs[i] for i in F])) if F else 1.0
                        CR = float(np.prod([cs[i] for i in range(N) if i not in Fs])) if len(F) < N else 1.0
                        I = info_fragment(C_all, CF, CR)
                        I2 = info_fragment(C_all, CR, CF)
                        L.check(abs(I + I2 - 2 * HS) < 1e-10, "heterogeneous complement identity")
                        L.check_monotone(CF, C_all, label="heterogeneous J^F >= J_phys")
                        Is.append(I)
                    if kd is None and np.mean(Is) >= (1 - DELTA) * HS - 1e-12:
                        kd = k
                kds.append(kd if kd is not None else float("nan"))
            rows.append([cmin, N, float(np.mean(Jps)), float(np.std(Jps)), float(np.nanmean(kds)), float(np.nanstd(kds))])
    out.table("6_heterogeneous", {**HEADER, "c_k": "uniform in [c_min, 1]", "seeds": "20 per point, seed = 1000 N + s (+7 for c_min 0.9)",
                                  "channel": "partial trace onto fragments (all if C(N,k) <= 1000, else 200 random)"},
              ["c_min", "N", "mean J_phys", "sd J_phys", "mean k_delta (fragment-averaged I)", "sd k_delta"], rows)
    out.say("All checks passed: closed forms, complement identity, k = 0 and k = N limits, monotonicity, Fuchs-van de Graaf.")
    out.say(L.SATURATION_NOTE)
    out.finish()


if __name__ == "__main__":
    main()
