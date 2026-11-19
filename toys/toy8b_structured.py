"""Toy 8b: structured webs (error-correcting codes, quantitative; a time crystal, qualitative).

Premise. A code spreads one logical question over many qubits so that no small part holds it: the
subset curve stays at J = 1 until k = d. Noise joins the alternatives; correction (syndrome records,
then a fix) splits them again, and each syndrome bit it records must be erased before reuse, at a cost
of at least kT ln 2. Without active correction every code here loses the thread at a rate set by the
noise. The known passive exception, a self-correcting memory at finite temperature, needs four spatial
dimensions (the 4D toric code); nothing in this toy models it.

Codes: 3- and 5-qubit repetition (bit-flip) codes, the [[5,1,3]] code (cyclic stabilizers XZZXI), Steane
[[7,1,3]]. Two questions per code: the Z question (|0_L> or |1_L>) and the X question (|+_L> or |-_L>).
Noise per step and per qubit: depolarising (X, Y, Z each with eps/3) or dephasing (Z with eps), reported
separately. Correction: with probability r per step, measure every stabilizer, store the syndrome in a
register (overwriting the last one) and apply the decoder's fix.

Method (exact, no sampling). Pauli noise and syndrome-only correction keep each state a mixture of
C_s L |q_L>, where s is the syndrome, C_s the decoder's fix and L one of I, X_L, Z_L, Y_L. The
probability table P(register, s, L) evolves exactly, is the same for both alternatives, and gives
  J over all qubits         sum_s 2 sqrt(a_s b_s)      (a, b: mass on the logical classes that keep or flip q)
  J with the register       sum_(reg, s) 2 sqrt(a b)
  J over the records        1 (the syndromes are independent of q; checked from the density matrices)
Density matrices are built from the table at snapshot times for the subset curves (exact, all subsets),
and a direct density-matrix simulation of the same channels checks the table for every code.

Time crystal (qualitative): the kicked Ising chain of Mi et al. (2022), parameters from memory. The
question is the initial value of the middle qubit; J over that qubit (k = 1) and over it with its two
neighbours (k = 3), with Z-dephasing trajectories.

Run: python toys/toy8b_structured.py            codes; time crystal at L = 12
     python toys/toy8b_structured.py --L20      adds the time crystal at L = 20 (slow)
     python toys/toy8b_structured.py --quick    checks and a small sweep
"""

import cmath
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

KB = 1.380649e-23
T_KELVIN = 300.0
STEP_S = 1e-6
EPS_LIST = [0.001, 0.003, 0.01, 0.03]
R_LIST = [0.0, 0.1, 0.5, 1.0]
T_MAX = 20000
SNAP_TIMES = [0, 10, 100, 1000]
NOISES = ["depolarizing", "dephasing"]

CODES = {
    "rep3": (["ZZI", "IZZ"], "XXX", "ZII"),
    "rep5": (["ZZIII", "IZZII", "IIZZI", "IIIZZ"], "XXXXX", "ZIIII"),
    "[[5,1,3]]": (["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"], "XXXXX", "ZZZZZ"),
    "Steane [[7,1,3]]": (["IIIXXXX", "IXXIIXX", "XIXIXIX", "IIIZZZZ", "IZZIIZZ", "ZIZIZIZ"], "XXXXXXX", "ZZZZZZZ"),
}
DISTANCE = {"rep3": "3 (X), 1 (Z)", "rep5": "5 (X), 1 (Z)", "[[5,1,3]]": "3", "Steane [[7,1,3]]": "3"}
QUESTIONS = {"Z": 2, "X": 1}       # logical-class bit that flips the answer: X_L part flips Z; Z_L part flips X


# ---------------------------------------------------------------- Pauli algebra on (x mask, z mask)
def pauli(s):
    n, x, z = len(s), 0, 0
    for j, ch in enumerate(s):
        b = 1 << (n - 1 - j)              # qubit j is bit n-1-j, so reshape axis j is qubit j
        x |= b if ch in "XY" else 0
        z |= b if ch in "ZY" else 0
    return (x, z)


def par(v):
    return bin(v).count("1") & 1


def sym(p, q):
    return par(p[0] & q[1]) ^ par(p[1] & q[0])


def mul(p, q):
    return (p[0] ^ q[0], p[1] ^ q[1])


class Code:
    def __init__(self, name):
        gens, xl, zl = CODES[name]
        self.name, self.n, self.m = name, len(xl), len(gens)
        self.g = [pauli(s) for s in gens]
        self.XL, self.ZL = pauli(xl), pauli(zl)
        for a in self.g:
            L.check(a[0] & a[1] == 0, "generators have no Y, so X^a Z^b is the operator itself")
            L.check(all(sym(a, b) == 0 for b in self.g) and sym(a, self.XL) == 0 and sym(a, self.ZL) == 0,
                    f"{name}: stabilizers commute with each other and the logicals")
        L.check(sym(self.XL, self.ZL) == 1, f"{name}: X_L and Z_L anticommute")
        self.css = all(p[0] == 0 or p[1] == 0 for p in self.g)
        self.C = self._decoder()
        self._tables()

    def syn(self, p):
        return sum(sym(p, g) << i for i, g in enumerate(self.g))

    def cls(self, p):                     # for p in the normaliser: 2 * (has X_L part) + (has Z_L part)
        return 2 * sym(p, self.ZL) + sym(p, self.XL)

    def _decoder(self):
        n, S = self.n, 1 << self.m
        C = [None] * S
        if self.css:                      # decode X and Z parts separately, minimum weight each
            Cx, Cz = {}, {}
            for w in range(n + 1):
                for qs in itertools.combinations(range(n), w):
                    mask = sum(1 << (n - 1 - j) for j in qs)
                    Cx.setdefault(self.syn((mask, 0)), mask)
                    Cz.setdefault(self.syn((0, mask)), mask)
            for sx, mx in Cx.items():
                for sz, mz in Cz.items():
                    C[sx ^ sz] = (mx, mz)
        else:                             # minimum weight over all Paulis, ties to X, then Z, then Y
            for w in range(n + 1):
                for qs in itertools.combinations(range(n), w):
                    for labs in itertools.product("XZY", repeat=w):
                        s = ["I"] * n
                        for j, c in zip(qs, labs):
                            s[j] = c
                        p = pauli("".join(s))
                        if C[self.syn(p)] is None:
                            C[self.syn(p)] = p
        L.check(all(c is not None and self.syn(c) == s for s, c in enumerate(C)), f"{self.name}: a fix for every syndrome")
        return C

    def _tables(self):
        n, S = self.n, 1 << self.m
        self.kappa = np.zeros((S, S), int)
        for s in range(S):
            for t in range(S):
                K = mul(mul(self.C[s ^ t], self.C[s]), self.C[t])
                L.check(self.syn(K) == 0, "fix products are in the normaliser")
                self.kappa[s, t] = self.cls(K)
        self.dest = {}
        sv, lv = np.arange(S)[:, None], np.arange(4)[None, :]
        for j in range(n):
            b = 1 << (n - 1 - j)
            for lab, p in (("X", (b, 0)), ("Y", (b, b)), ("Z", (0, b))):
                sF = self.syn(p)
                NF = mul(self.C[sF], p)
                LF = self.cls(NF)
                self.dest[(j, lab)] = ((sv ^ sF) * 4 + (lv ^ LF ^ self.kappa[:, sF][:, None])).reshape(-1)


# ---------------------------------------------------------------- exact evolution of P(register, s, L)
def weights(noise, eps):
    return {"X": eps / 3, "Y": eps / 3, "Z": eps / 3} if noise == "depolarizing" else {"Z": eps}


def step(code, P, w, eps, r):
    for j in range(code.n):
        new = (1 - eps) * P
        for lab, wl in w.items():
            new[:, code.dest[(j, lab)]] += wl * P
        P = new
    if r > 0:
        Q = P.sum(axis=0).reshape(-1, 4)
        P = (1 - r) * P
        P[:, 0:4] += r * Q                # register <- s, syndrome -> 0, logical class kept
    return P


def joins(P, S):
    Pr = P.reshape(S, S, 4)
    out = {}
    for q, bit in QUESTIONS.items():
        a = Pr[..., [l for l in range(4) if not l & bit]].sum(-1)
        b = Pr[..., [l for l in range(4) if l & bit]].sum(-1)
        out[q] = (float(2 * np.sum(np.sqrt(a.sum(0) * b.sum(0)))), float(2 * np.sum(np.sqrt(a * b))))
    return out


def code_job(args):
    name, noise, eps, r, t_max, snaps = args
    code = Code(name)
    S = 1 << code.m
    P = np.zeros((S, 4 * S))
    P[0, 0] = 1.0
    w = weights(noise, eps)
    life = {}
    kept = {}
    for t in range(0, t_max + 1):
        if t > 0:
            P = step(code, P, w, eps, r)
        if t in snaps:
            kept[t] = P.sum(axis=0).reshape(S, 4).copy()
        J = joins(P, S)
        for q in QUESTIONS:
            for i, lab in enumerate(("qubits", "register")):
                if (q, lab) not in life and J[q][i] >= 0.5:
                    life[(q, lab)] = t
        if len(life) == 4 and t >= max(snaps):
            break
    return (name, noise, eps, r), life, kept, joins(P, S), t


# ---------------------------------------------------------------- density matrices from the table
class States:
    def __init__(self, code):
        self.code = code
        n = code.n
        self.N = 1 << n
        self.idx = np.arange(self.N)
        self.pc = np.array([bin(i).count("1") & 1 for i in range(self.N)])
        v = np.zeros(self.N, complex)
        v[0] = 1
        for g in code.g:
            v = (v + self.apply(g, v)) / 2
        v = (v + self.apply(code.ZL, v)) / 2
        L.check(np.linalg.norm(v) > 0.1, "the projected |0..0> is non-zero")
        z = v / np.linalg.norm(v)
        o = self.apply(code.XL, z)
        for g in code.g:
            L.check(np.allclose(self.apply(g, z), z) and np.allclose(self.apply(g, o), o), f"{code.name}: code states are stabilized")
        L.check(np.allclose(self.apply(code.ZL, z), z) and np.allclose(self.apply(code.ZL, o), -o), f"{code.name}: Z_L eigenstates")
        self.psi = {"Z": (z, o), "X": ((z + o) / math.sqrt(2), (z - o) / math.sqrt(2))}

    def apply(self, p, v):                # X^x Z^z v, phase dropped (exact for operators without Y)
        v = v * (1 - 2 * self.pc[self.idx & p[1]])[(slice(None),) + (None,) * (v.ndim - 1)]
        return v[self.idx ^ p[0]]

    def columns(self, Q, q, a):
        """Weighted vectors W with rho = W W^dag for alternative a of question q."""
        code, cols = self.code, []
        for s in range(Q.shape[0]):
            for l in range(4):
                if Q[s, l] <= 0:
                    continue
                Lop = mul(code.XL if l & 2 else (0, 0), code.ZL if l & 1 else (0, 0))
                cols.append(math.sqrt(Q[s, l]) * self.apply(code.C[s], self.apply(Lop, self.psi[q][a])))
        return np.array(cols).T

    def reduced(self, W, K):
        n = self.code.n
        rest = [k for k in range(n) if k not in K]
        A = np.transpose(W.reshape([2] * n + [-1]), list(K) + rest + [n]).reshape(2 ** len(K), -1)
        return A @ A.conj().T

    def syndrome_probs(self, W):
        code, out = self.code, []
        for s in range(1 << code.m):
            V = W.copy()
            for i, g in enumerate(code.g):
                V = (V + (-1) ** ((s >> i) & 1) * self.apply(g, V)) / 2
            out.append(float(np.sum(np.abs(V) ** 2)))
        return np.array(out)


def subset_curve(st, Q, q):
    n = st.code.n
    W0, W1 = st.columns(Q, q, 0), st.columns(Q, q, 1)
    curve = []
    for k in range(1, n + 1):
        curve.append(min(L.fidelity(st.reduced(W0, K), st.reduced(W1, K)) for K in itertools.combinations(range(n), k)))
    kh = next((k + 1 for k, v in enumerate(curve) if v <= 0.5), None)
    return curve, kh


def curve_job(args):
    name, noise, eps, r, t, Q = args
    st = States(Code(name))
    res = {}
    for q in QUESTIONS:
        res[q] = subset_curve(st, Q, q)
        p0 = st.syndrome_probs(st.columns(Q, q, 0))
        p1 = st.syndrome_probs(st.columns(Q, q, 1))
        L.check(np.allclose(p0, p1, atol=1e-12), "the syndrome records are the same for both alternatives (J over records = 1)")
        Jall = L.fidelity(st.reduced(st.columns(Q, q, 0), list(range(st.code.n))), st.reduced(st.columns(Q, q, 1), list(range(st.code.n))))
        a = Q[:, [l for l in range(4) if not l & QUESTIONS[q]]].sum(-1)
        b = Q[:, [l for l in range(4) if l & QUESTIONS[q]]].sum(-1)
        L.check(abs(Jall - 2 * np.sum(np.sqrt(a * b))) < 1e-7, f"J over all qubits from the density matrices matches the table ({Jall})")
    return (name, noise, eps, r, t), res


# ---------------------------------------------------------------- direct density-matrix check
def dense_pauli(st, p):
    M = np.zeros((st.N, st.N))
    M[st.idx ^ p[0], st.idx] = 1 - 2 * st.pc[st.idx & p[1]]
    return M


def direct_check(name, noise="depolarizing", eps=0.05, r=0.5, steps=8, with_register=False):
    code = Code(name)
    st = States(code)
    n, S = code.n, 1 << code.m
    one = {lab: [dense_pauli(st, pauli("I" * j + lab + "I" * (n - 1 - j))) for j in range(n)] for lab in "XYZ"}
    G = [dense_pauli(st, g) for g in code.g]
    Pi = []
    for s in range(S):
        M = np.eye(st.N)
        for i in range(code.m):
            M = M @ (np.eye(st.N) + (-1) ** ((s >> i) & 1) * G[i]) / 2
        Pi.append(dense_pauli(st, code.C[s]) @ M)
    w = weights(noise, eps)
    P = np.zeros((S, 4 * S)); P[0, 0] = 1.0
    worst = 0.0
    for q in QUESTIONS:
        blocks = [[np.zeros((st.N, st.N), complex) for _ in range(S)] for _ in (0, 1)]
        for a in (0, 1):
            blocks[a][0] = np.outer(st.psi[q][a], st.psi[q][a].conj())
        Pq = P.copy()
        for _ in range(steps):
            for a in (0, 1):
                B = blocks[a]
                for j in range(n):
                    B = [(1 - eps) * X + sum(wl * one[lab][j] @ X @ one[lab][j].T for lab, wl in w.items()) for X in B]
                tot = sum(B)
                B = [(1 - r) * X for X in B]
                for s in range(S):
                    B[s] = B[s] + r * Pi[s] @ tot @ Pi[s].T
                blocks[a] = B
            Pq = step(code, Pq, w, eps, r)
            J = joins(Pq, S)[q]
            Jq = L.fidelity(sum(blocks[0]), sum(blocks[1]))
            worst = max(worst, abs(Jq - J[0]))
            if with_register:
                Jr = sum(L.fidelity(blocks[0][s], blocks[1][s]) for s in range(S))
                worst = max(worst, abs(Jr - J[1]))
    L.check(worst < 1e-7, f"{name}: direct density-matrix simulation matches the table (worst difference {worst:.1e})")
    return worst


# ---------------------------------------------------------------- exporting cuts (revision 4.6, section 2 item 3)
PSI_L = np.array([math.cos(math.pi / 8), cmath.exp(1j * math.pi / 4) * math.sin(math.pi / 8)])
_X2 = np.array([[0, 1], [1, 0]], complex)
_Z2 = np.diag([1.0, -1.0]).astype(complex)
LVEC = [PSI_L, _Z2 @ PSI_L, _X2 @ PSI_L, _X2 @ _Z2 @ PSI_L]           # index l = 2 * (X_L part) + (Z_L part)
LPROJ = np.array([np.outer(v, v.conj()) for v in LVEC])


def logical_blocks(P, S, with_register):
    """The state for a fixed logical state psi is block diagonal: one 2 x 2 block per syndrome s (and per register
    value, if the register is included), spanned by C_s|0_L> and C_s|1_L>."""
    Pr = P.reshape(S, S, 4)
    if not with_register:
        Pr = Pr.sum(0, keepdims=True)
    return np.einsum("rsl,lij->rsij", Pr, LPROJ).reshape(-1, 2, 2)


def _block_entropy(B):
    w = np.linalg.eigvalsh((B + B.conj().swapaxes(-1, -2)) / 2).reshape(-1)
    w = w[w > 1e-15]
    return float(-np.sum(w * np.log2(w)))


def block_holevo(B0, B1):
    return _block_entropy((B0 + B1) / 2) - (_block_entropy(B0) + _block_entropy(B1)) / 2


def ledger_join(code, sE, lE, r, T):
    """Noise-free case: enumerate the steps at which correction ran (a record both alternatives share). Given that,
    each alternative is pure: C_s L psi, the register, and every overwritten register value (exported as heat).
    Returns J over all of it at each step."""
    out = []
    for t in range(T + 1):
        J = 0.0
        for h in itertools.product((0, 1), repeat=t):
            ph = float(np.prod([r if x else 1 - r for x in h])) if t else 1.0
            if ph == 0:
                continue
            st = []
            for s, l in ((0, 0), (sE, lE)):
                reg, ledger = 0, []
                for x in h:
                    if x:
                        ledger.append(reg)
                        reg, s = s, 0
                st.append((s, l, reg, tuple(ledger)))
            (s0, l0, r0, g0), (s1, l1, r1, g1) = st
            if s0 == s1 and r0 == r1 and g0 == g1:
                J += ph * abs(np.vdot(LVEC[l0], LVEC[l1]))
        out.append(J)
    return out


def export_job(args):
    name, err, eps, r, T = args
    code = Code(name)
    S = 1 << code.m
    p = pauli(err)
    sE = code.syn(p)
    lE = code.cls(mul(code.C[sE], p))
    P = [np.zeros((S, 4 * S)) for _ in (0, 1)]
    P[0][0, 0] = 1.0
    P[1][0, sE * 4 + lE] = 1.0
    w = weights("depolarizing", eps)
    rows = []
    for t in range(T + 1):
        if t > 0:
            P = [step(code, X, w, eps, r) for X in P]
        Bq = [logical_blocks(X, S, False) for X in P]
        Br = [logical_blocks(X, S, True) for X in P]
        rows.append((t, float(np.sum(L.fidelity(*Bq))), float(np.sum(L.fidelity(*Br))), block_holevo(*Bq), block_holevo(*Br)))
    everything = ledger_join(code, sE, lE, r, T) if eps == 0 else None
    return (name, err, eps, r), rows, everything, sE != 0


def export_errors(name):
    n = len(CODES[name][1])
    errs = ["X" + "I" * (n - 1), "Z" + "I" * (n - 1)]
    return errs + (["Y" + "I" * (n - 1)] if name == "rep3" else [])


# ---------------------------------------------------------------- time crystal
def crystal_job(args):
    L_, g, p, T, ntraj, seed, rec_times = args
    rng = np.random.default_rng(seed)
    N = 1 << L_
    idx = np.arange(N)
    bits = ((idx[:, None] >> (L_ - 1 - np.arange(L_))[None, :]) & 1).astype(np.int8)
    z = 1 - 2 * bits.astype(float)
    h = rng.uniform(-math.pi, math.pi, L_)
    phi = rng.uniform(-1.5 * math.pi, -0.5 * math.pi, L_ - 1)
    D = np.exp(-0.5j * (z @ h) - 0.25j * np.sum(phi[None, :] * z[:, :-1] * z[:, 1:], axis=1))
    c, s = math.cos(math.pi * g / 2), math.sin(math.pi * g / 2)
    j0 = L_ // 2
    s0 = int(rng.integers(0, N))
    starts = (s0, s0 ^ (1 << (L_ - 1 - j0)))
    if p == 0:
        ntraj = 1
    rho1 = {t: [np.zeros((2, 2), complex), np.zeros((2, 2), complex)] for t in rec_times}
    rho3 = {t: [np.zeros((8, 8), complex), np.zeros((8, 8), complex)] for t in rec_times}
    zall = {t: np.zeros(L_) for t in rec_times}
    glob = 0.0
    nb = [j0 - 1, j0, j0 + 1]

    def red(psi, K):
        rest = [k for k in range(L_) if k not in K]
        A = np.transpose(psi.reshape([2] * L_), K + rest).reshape(2 ** len(K), -1)
        return A @ A.conj().T

    for _ in range(ntraj):
        psis = []
        for st in starts:
            v = np.zeros(N, complex); v[st] = 1
            psis.append(v)
        for t in range(0, T + 1):
            if t > 0:
                for a in (0, 1):
                    v = psis[a].reshape([2] * L_)
                    for j in range(L_):
                        v = c * v - 1j * s * np.flip(v, axis=j)
                    psis[a] = D * v.reshape(-1)
                if p > 0:
                    m = (rng.random(L_) < p).astype(np.int8)
                    sign = 1 - 2 * ((bits @ m) & 1)
                    psis = [sign * v for v in psis]
                if p == 0:
                    glob = max(glob, abs(np.vdot(psis[0], psis[1])))
            if t in rec_times:
                for a in (0, 1):
                    rho1[t][a] += red(psis[a], [j0]) / ntraj
                    rho3[t][a] += red(psis[a], nb) / ntraj
                zall[t] += (np.abs(psis[0]) ** 2 @ z) * ((-1) ** t) * z[s0] / ntraj
    out = []
    for t in rec_times:
        J1 = L.fidelity(rho1[t][0], rho1[t][1])
        J3 = L.fidelity(rho3[t][0], rho3[t][1])
        out.append((t, J1, J3, float(zall[t][j0]), float(np.mean(zall[t]))))
    return (L_, g, p), out, glob


# ---------------------------------------------------------------- main
def run_checks(out):
    for name in CODES:
        st = States(Code(name))
        Q = np.zeros((1 << st.code.m, 4)); Q[0, 0] = 1
        n = st.code.n
        for q in QUESTIONS:
            curve, kh = subset_curve(st, Q, q)
            if name in ("[[5,1,3]]", "Steane [[7,1,3]]"):
                L.check(all(abs(v - 1) < 1e-9 for v in curve[:2]), f"{name}, {q} question: J = 1 for k < d = 3 ({curve[:2]})")
                L.check(abs(curve[-1]) < 1e-9, f"{name}: all qubits hold the answer")
            elif q == "Z":
                L.check(abs(curve[0]) < 1e-9, f"{name}: one qubit already holds the Z answer")
            else:
                L.check(all(abs(v - 1) < 1e-9 for v in curve[:-1]) and abs(curve[-1]) < 1e-9, f"{name}: X answer only in all qubits")
        out.say(f"{name}, no noise: subset curve Z question {[round(float(v), 3) for v in subset_curve(st, Q, "Z")[0]]}, "
                f"X question {[round(float(v), 3) for v in subset_curve(st, Q, "X")[0]]} (k = 1..{n}).")
    worst = max(direct_check(n_, with_register=(n_ != "Steane [[7,1,3]]")) for n_ in CODES)
    worst = max(worst, max(direct_check(n_, noise="dephasing", eps=0.08, r=0.3, steps=5) for n_ in ("rep3", "[[5,1,3]]")))
    out.say(f"A direct density-matrix simulation of noise and correction matches the exact table for every code "
            f"(worst difference in J {worst:.1e}; with the register too, except Steane).")
    for name in ("[[5,1,3]]", "Steane [[7,1,3]]", "rep3"):
        _, rows, every, seen = export_job((name, "X" + "I" * (len(CODES[name][1]) - 1), 0.0, 1.0, 3))
        L.check(np.allclose([x[1] for x in rows], [0, 1, 1, 1], atol=1e-6), f"{name}: correction re-joins the qubits at the first step")
        L.check(np.allclose([x[2] for x in rows], [0, 0, 1, 1], atol=1e-6), f"{name}: qubits with the register re-join only when it is overwritten")
        L.check(max(every) < 1e-12, f"{name}: over everything the cut stays made")
    _, rows, every, seen = export_job(("rep3", "ZII", 0.0, 1.0, 3))
    z = abs(np.vdot(PSI_L, _Z2 @ PSI_L))
    L.check(not seen and all(abs(x[1] - z) < 1e-6 and abs(x[2] - z) < 1e-6 for x in rows) and all(abs(e - z) < 1e-12 for e in every),
            "rep3: an error the code cannot see stays a cut in the qubits")
    out.say("Exporting cuts (no noise, correction every step): for a detected error, J over the qubits goes 0 -> 1 at the first "
            "correction, J over the qubits and register goes 0 -> 1 when the register is overwritten, and J over everything "
            f"(including each overwritten register value) stays 0. An error the code cannot see (Z on rep3) stays at J = {z:.4f} everywhere.")
    res = crystal_job((6, 1.0, 0.0, 20, 1, 7, list(range(21))))
    L.check(all(abs(r_[1]) < 1e-9 and abs(r_[4] - 1) < 1e-9 for r_ in res[1]), "g = 1, no noise: exact period doubling, J over the middle qubit = 0")
    L.check(res[2] < 1e-9, "no noise: the two alternatives stay orthogonal (global J = 0)")
    out.say("Time crystal, g = 1 and no noise (L = 6, 20 cycles): every qubit's Z value flips each cycle exactly; "
            "the middle qubit alone keeps the answer (J = 0) and the global overlap stays 0.")


def main(quick=False, with_L20=False):
    out = L.Output("toy8b", "Toy 8b: structured webs (codes and a time crystal)")
    out.flag("4D exception (adopted 17 Nov)", "stated in the premise: without active correction every code here decays; "
             "the known passive exception is the 4D toric code, not modelled", "spec")
    out.flag("correction as a partial channel", "with probability r per step: measure all stabilizers, overwrite the register, "
             "apply the decoder's fix; the toy evolves the exact average, so no realisations are needed", "spec allowed either; exact is cheaper")
    out.flag("exact table P(register, s, L)", "Pauli noise and syndrome-only correction keep every state a mixture of C_s L |q_L>; "
             "checked against a direct density-matrix simulation for every code", "makes Steane cheap")
    out.flag("decoders", "CSS codes: X and Z parts decoded separately, minimum weight; [[5,1,3]]: minimum weight (unique, "
             "perfect code); repetition codes fix only bit flips", "standard choices; ties do not arise")
    out.flag("two questions", "Z question (|0_L> vs |1_L>) and X question (|+_L> vs |-_L>), reported separately",
             "a code protects the two differently; the repetition code hides the X answer but exposes the Z answer")
    out.flag("noise parameters", f"eps per qubit per step in {EPS_LIST}; r in {R_LIST}; up to {T_MAX} steps", "spans no correction to every step")
    out.flag("lifetime", "first step with J >= 0.5, over the qubits and over the qubits with the register; '>' if not reached",
             "spec; J = 0.5 is a logical error probability of about 0.067")
    out.flag("heat", f"n_stab * r * kT ln 2 per step at {T_KELVIN:g} K; step time {STEP_S * 1e6:g} microsecond (flagged); "
             "Landauer's bound for erasing each syndrome bit, a lower bound only", "spec")
    out.flag("subset curves", f"exact minimum over all subsets of each size, at steps {SNAP_TIMES}, eps = 0.01, r = 0 and 1",
             "every subset is cheap for n <= 7")
    out.flag("time crystal parameters", "U_F = exp(-i/2 sum h_j Z_j) exp(-i/4 sum phi_j Z_j Z_j+1) exp(-i pi g/2 sum X_j); "
             "h_j in [-pi, pi], phi_j in [-1.5 pi, -0.5 pi], g in {1, 0.97, 0.9, 0.6}; open chain; from memory of Mi et al. 2022",
             "not checked against the paper (no access here)")
    out.flag("time crystal question and noise", "the initial value of the middle qubit (a random bitstring, or it with the middle "
             "qubit flipped); Z-dephasing with probability p per qubit per cycle, trajectories averaged; global J not computed with noise",
             "Z noise commutes with the period-doubled Z pattern at g = 1, so it bites only when g < 1")
    out.flag("exporting cuts (revision 4.6, section 2 item 3)", "question: was an error E (X or Z on qubit 0; Y too on rep3) "
             "injected just before step 1? The logical state is fixed at a generic psi; depolarising noise eps in {0, 0.01}; r in {1, 0.5}. "
             "J over the qubits, over the qubits with the register, and (eps = 0 only) over everything: qubits, register and every "
             "overwritten register value", "with noise, 'everything' includes the noise's own environment and is 0 by unitarity; not computed")
    run_checks(out)

    names = ["rep3", "[[5,1,3]]"] if quick else list(CODES)
    eps_l = [0.01, 0.03] if quick else EPS_LIST
    r_l = [0.0, 1.0] if quick else R_LIST
    t_max = 1500 if quick else T_MAX
    snaps = [0, 10, 100] if quick else SNAP_TIMES
    jobs = [(nm, nz, e, r, t_max, snaps) for nm in names for nz in NOISES for e in eps_l for r in r_l]
    t0 = time.time()
    with Pool(workers()) as pool:
        res = pool.map(code_job, jobs, chunksize=1)
        cjobs = [(k[0], k[1], k[2], k[3], t, kept[t]) for k, life, kept, _, _ in res
                 if k[2] == 0.01 and k[3] in (0.0, 1.0) for t in snaps if t in kept]
        curves = pool.map(curve_job, cjobs, chunksize=1)
    out.say(f"Codes: {len(jobs)} runs and {len(cjobs)} subset-curve snapshots in {time.time() - t0:.0f} s.")

    hdr = {"event": "encoding (the logical qubit spread over n qubits)", "question": "Z: |0_L> or |1_L>; X: |+_L> or |-_L>",
           "prior": "uniform", "records": "syndrome bits, the last kept in a register"}
    kTln2 = KB * T_KELVIN * math.log(2)
    rows1 = []
    for (nm, nz, e, r), life, kept, Jend, tend in res:
        code = Code(nm)
        heat_step = code.m * r * kTln2
        for q in QUESTIONS:
            lq, lr = life.get((q, "qubits")), life.get((q, "register"))
            rows1.append([nm, DISTANCE[nm], nz, e, r, q,
                          lq if lq is not None else f"> {tend}", lr if lr is not None else f"> {tend}",
                          heat_step / STEP_S, heat_step * (lq if lq is not None else tend)])
    out.table("1_lifetimes", hdr, ["code", "distance", "noise", "eps", "r", "question", "lifetime, qubits (steps)",
                                   "lifetime, qubits + register", "heat rate (W, Landauer)", "heat over lifetime (J)"], rows1,
              note="J over the syndrome records alone is 1 at every step (checked): records never hold the answer.")
    rows2 = []
    for (nm, nz, e, r, t), cv in sorted(curves, key=lambda x: (x[0][0], x[0][1], x[0][3], x[0][4])):
        for q in QUESTIONS:
            curve, kh = cv[q]
            rows2.append([nm, nz, e, r, t, q] + [round(v, 6) for v in curve] + [""] * (7 - len(curve)) + [kh if kh else "none"])
    out.table("2_subset_curves", hdr, ["code", "noise", "eps", "r", "step", "question"] + [f"k = {k}" for k in range(1, 8)] + ["k_half"], rows2)

    ejobs = [(nm, e, eps, r, 3 if quick else 6) for nm in names for e in export_errors(nm)
             for eps in (0.0, 0.01) for r in (1.0, 0.5)]
    with Pool(workers()) as pool:
        eres = pool.map(export_job, ejobs, chunksize=1)
    rows4 = []
    for (nm, e, eps, r), rows, every, seen in eres:
        for i, (t, Jq, Jr, cq, cr) in enumerate(rows):
            rows4.append([nm, e, "yes" if seen else "no", eps, r, t, Jq, Jr, every[i] if every is not None else "", cq, cr])
    out.table("4_exporting_cuts", {"event": "an error E on qubit 0, just before step 1, or no error",
                                   "question": "was E injected? (the logical state is fixed at a generic psi)",
                                   "prior": "uniform", "noise": "depolarising, eps per qubit per step",
                                   "records": "qubits; the syndrome register; every overwritten register value (exported as heat)"},
              ["code", "error E", "syndrome sees E", "eps", "r", "step", "J over qubits", "J over qubits + register",
               "J over everything (eps = 0 only)", "chi bits, qubits", "chi bits, qubits + register"], rows4,
              note="Revision 4.6, section 2 item 3: correction moves the cut out of the qubits, then out of the register; over "
                   "everything it stays made. For the logical question (table 1) correction does the opposite: it keeps the qubits cut.")
    for (nm, e, eps, r), rows, every, seen in eres:
        if eps == 0.01 and r == 1.0:
            out.say(f"Exporting cuts, {nm}, {e}, eps = 0.01, r = 1: J over qubits {', '.join(f'{x[1]:.3f}' for x in rows)}; "
                    f"with register {', '.join(f'{x[2]:.3f}' for x in rows)} (steps 0..{rows[-1][0]})")

    Ls = [6] if quick else [12] + ([20] if with_L20 else [])
    gs = [1.0, 0.6] if quick else [1.0, 0.97, 0.9, 0.6]
    ps = [0.0, 0.02] if quick else [0.0, 0.005, 0.02]
    T = 20 if quick else 100
    rec = [t for t in (0, 1, 2, 3, 4, 5, 10, 20, 50, 100) if t <= T]
    cj = []
    for L_ in Ls:
        nreal = 3 if quick else (5 if L_ >= 20 else 20)
        ntraj = 4 if quick else (5 if L_ >= 20 else 20)
        for g in gs:
            for p in ps:
                for k in range(nreal):
                    cj.append((L_, g, p, T, ntraj, 1000 * L_ + 100 * k + int(1000 * g) + int(1e4 * p), rec))
    t0 = time.time()
    with Pool(workers()) as pool:
        cres = pool.map(crystal_job, cj, chunksize=1)
    out.say(f"Time crystal: {len(cj)} disorder realisations in {time.time() - t0:.0f} s.")
    agg = {}
    for key, rows, glob in cres:
        agg.setdefault(key, []).append((rows, glob))
    rows3 = []
    for (L_, g, p), lst in sorted(agg.items()):
        for i, t in enumerate(rec):
            vals = np.array([[r_[0][i][1], r_[0][i][2], r_[0][i][3], r_[0][i][4]] for r_ in lst])
            gl = max(r_[1] for r_ in lst) if p == 0 else float("nan")
            rows3.append([L_, g, p, t] + [round(float(x), 4) for x in vals.mean(0)] + [gl])
    out.table("3_time_crystal", {"event": "the initial bitstring", "question": "the initial value of the middle qubit",
                                 "prior": "uniform", "records": "the chain itself"},
              ["L", "g", "p", "cycle", "J over the middle qubit (k = 1)", "J over it and its neighbours (k = 3)",
               "middle-qubit autocorrelation (-1)^t <Z(t)> Z(0)", "chain-mean autocorrelation", "global J (no noise; max over cycles)"], rows3,
              note="Qualitative: averages over disorder realisations; J is averaged after it is computed per realisation.")
    for (L_, g, p), lst in sorted(agg.items()):
        J1 = np.mean([r_[0][-1][1] for r_ in lst])
        out.say(f"Time crystal L = {L_}, g = {g}, p = {p}: J over the middle qubit after {T} cycles = {J1:.3f}.")
    out.say(L.SATURATION_NOTE)
    out.finish()


if __name__ == "__main__":
    main(quick="--quick" in sys.argv, with_L20="--L20" in sys.argv)
