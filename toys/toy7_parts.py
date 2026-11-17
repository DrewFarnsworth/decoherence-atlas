"""Toy 7: whose parts? Does the pointer basis fix the choice of parts?

Toy 4's graphs G1 and G3 (same H, disorder W, time t). The alternatives are "which basis state
took the excitation"; a candidate domain R is a set of basis states, registered by projection onto
their span plus 'outside' (toy 4's Phi^trace, in the new basis).

How the rest of the universe couples (the later events in the parts' future, modelled
phenomenologically). Local dephasing presupposes sites as parts, so every basis runs under three
couplings, each with rate gamma:
    local       sqrt(gamma) |k><k| for each site (as toy 4)
    collective  sqrt(gamma) P_c for each cluster c (projector onto its sites)
    random      sqrt(gamma) h_m, N Hermitian h_m from the Gaussian unitary ensemble, normalised; 10 seeds

Bases: site (must reproduce toy 4), exciton (eigenstates of H), pointer (predictability sieve:
the orthonormal basis whose members, each evolved alone to t, lose least purity), 20 Haar-random.
Priors: uniform, and thermal p_a ~ exp(-<a|H|a>/kT), kT = V.

Counts against the rule: pointer-basis domains no more stable than random-basis domains; domains
that reverse rank when the prior changes; or a pointer basis that does not follow the coupling.

Run: python toys/toy7_parts.py           (reduced sweep, see FLAGS)
     python toys/toy7_parts.py --quick   (checks and one point per coupling)
"""

import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge_lib as L  # noqa: E402
import toy4_graph_domains as T4  # noqa: E402

N_RANDOM_BASES = 20
RANDOM_BASIS_SEED = 7000
RANDOM_COUPLING_SEEDS = list(range(10))
RANDOM_COUPLING_BASE = 7100
V_LIST = [0.05, 0.2]
W_LIST = [0.0, 0.5]
G_LIST = T4.G_LIST
T_LIST = T4.T_LIST
RANDOM_G_LIST = [1.0, 10.0]
RANDOM_T_LIST = [3.0, 30.0]
KT = 1.0
CLUSTERS = [list(range(0, 4)), list(range(4, 8))]


# ---------------------------------------------------------------- couplings
def gue(n, rng):
    A = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / math.sqrt(2)
    h = (A + A.conj().T) / 2
    return h / np.linalg.norm(h)             # unit Frobenius norm, like a site projector


def coupling_ops(kind, n, seed=0):
    if kind == "local":
        return [np.diag(np.eye(n)[k]).astype(complex) for k in range(n)]
    if kind == "collective":
        ops = []
        for c in CLUSTERS:
            P = np.zeros((n, n), complex)
            P[c, c] = 1.0
            ops.append(P)
        return ops
    if kind == "random":
        rng = np.random.default_rng(RANDOM_COUPLING_BASE + seed)
        return [gue(n, rng) for _ in range(n)]
    raise ValueError(kind)


def coupling_eigenbasis(kind, ops):
    """The basis the coupling itself singles out (see FLAGS for the random case)."""
    n = ops[0].shape[0]
    if kind == "local":
        return np.eye(n, dtype=complex), None
    if kind == "collective":
        return None, ops                           # degenerate within clusters: compare with cluster subspaces
    w, U = np.linalg.eigh(sum(h @ h for h in ops))
    return U, None


def liouvillian(H, ops, gamma):
    n = H.shape[0]
    I = np.eye(n)
    Ls = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for A in ops:
        c = math.sqrt(gamma) * A
        cdc = c.conj().T @ c
        Ls = Ls + np.kron(c.conj(), c) - 0.5 * np.kron(I, cdc) - 0.5 * np.kron(cdc.T, I)
    return Ls


def propagator(H, ops, gamma, t):
    return expm(liouvillian(H, ops, gamma) * t)


# ---------------------------------------------------------------- bases
def haar(n, rng):
    Z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / math.sqrt(2)
    Q, R = np.linalg.qr(Z)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def evolve_basis(P, U):
    n = U.shape[0]
    out = []
    for a in range(n):
        r0 = np.outer(U[:, a], U[:, a].conj())
        out.append((P @ r0.flatten(order="F")).reshape(n, n, order="F"))
    return np.array(out)


def entropy_sum(P, U):
    recs = evolve_basis(P, U)
    return float(np.sum(1 - np.real(np.einsum("aij,aji->a", recs, recs))))


def pointer_basis(P, n, starts, iters=300):
    """Minimise summed linear entropy over U = U0 expm(iX); best result over the starting bases."""
    iu = np.triu_indices(n, 1)
    def unitary(x, U0):
        X = np.zeros((n, n), complex)
        X[iu] = x[: len(iu[0])] + 1j * x[len(iu[0]): 2 * len(iu[0])]
        X = X + X.conj().T
        X[np.diag_indices(n)] = x[2 * len(iu[0]):]
        return U0 @ expm(1j * X)
    best = None
    for U0 in starts:
        res = minimize(lambda x: entropy_sum(P, unitary(x, U0)), np.zeros(n * n), method="L-BFGS-B", options={"maxiter": iters})
        if best is None or res.fun < best[1].fun:
            best = (unitary(res.x, U0), res)
    return best


def in_basis(recs_site, U):
    return np.einsum("ji,ajk,kl->ail", U.conj(), recs_site, U)


def domains_for(recs_basis):
    return T4.analyse(recs_basis, None, ["trace"])["trace"]


def jaccard(a, b):
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def basis_overlap(U, Up):
    """Mean over pointer vectors of the best squared overlap with U's vectors."""
    return float(np.mean(np.max(np.abs(U.conj().T @ Up) ** 2, axis=0)))


def subspace_overlap(projs, Up):
    """Mean over pointer vectors of the largest weight inside one cluster subspace."""
    return float(np.mean([max(np.real(p.conj() @ P @ p) for P in projs) for p in Up.T]))


# ---------------------------------------------------------------- one point
def run_point(gname, v, W, gamma, t, kind="local", cseed=0):
    V, _ = T4.graph(gname, v)
    n = V.shape[0]
    H = np.diag(T4.disorder(n, W)) + V
    ops = coupling_ops(kind, n, cseed)
    P = propagator(H, ops, gamma, t)
    _, Uex = np.linalg.eigh(H)
    Uex = Uex.astype(complex)
    Ucpl, projs = coupling_eigenbasis(kind, ops)
    starts = [np.eye(n, dtype=complex), Uex] + ([Ucpl] if Ucpl is not None and kind == "random" else [])
    Up, opt = pointer_basis(P, n, starts)
    rng = np.random.default_rng(RANDOM_BASIS_SEED)
    bases = {"site": np.eye(n, dtype=complex), "exciton": Uex, "pointer": Up}
    for k in range(N_RANDOM_BASES):
        bases[f"random{k}"] = haar(n, rng)
    out = {}
    for name, U in bases.items():
        L.check(np.max(np.abs(U.conj().T @ U - np.eye(n))) < 1e-10, f"basis {name} unitary")
        res = domains_for(in_basis(evolve_basis(P, U), U))
        energies = np.real(np.einsum("ia,ij,ja->a", U.conj(), H, U))
        prior = np.exp(-(energies - energies.min()) / KT); prior /= prior.sum()
        dom = set(res["A"]) | set(res["B"])
        out[name] = {"A": res["A"], "B": res["B"],
                     "sig_u": {R: res["sigma"][R] for R in dom},
                     "sig_t": {R: L.sigma_of(res["J"][R], R, weights=prior) for R in dom}}
    ov = {"site": basis_overlap(np.eye(n), Up), "exciton": basis_overlap(Uex, Up)}
    ov["coupling"] = subspace_overlap(projs, Up) if projs is not None else basis_overlap(Ucpl, Up)
    return out, ov, opt


def points(quick):
    if quick:
        return [("local", 0, ("G1", 0.2, 0.0, 1.0, 3.0)), ("collective", 0, ("G1", 0.2, 0.0, 1.0, 3.0)),
                ("random", 0, ("G1", 0.2, 0.0, 1.0, 3.0))]
    pts = []
    for kind in ("local", "collective"):
        pts += [(kind, 0, (g, v, W, ga, t)) for g in ("G1", "G3") for v in V_LIST for W in W_LIST for ga in G_LIST for t in T_LIST]
    pts += [("random", s, (g, v, W, ga, t)) for s in RANDOM_COUPLING_SEEDS for g in ("G1", "G3") for v in V_LIST
            for W in W_LIST for ga in RANDOM_G_LIST for t in RANDOM_T_LIST]
    return pts


def main(quick=False):
    out = L.Output("toy7", "Toy 7: whose parts?")
    out.flag("reduced sweep", f"G1 and G3; v in {V_LIST}; W in {W_LIST}; local and collective couplings at every gamma and t; "
             f"random coupling at gamma in {RANDOM_G_LIST}, t in {RANDOM_T_LIST}", "the full toy-4 sweep times 23 bases and 12 couplings would take hours")
    out.flag("pointer optimiser", "L-BFGS-B on the Hermitian generator of U = U0 expm(iX), started from the site and exciton bases "
             "(and the coupling's basis for random coupling), 300 iterations, best kept; convergence printed per point", "the spec asks to flag it")
    out.flag("GUE normalisation", "each h_m scaled to unit Frobenius norm (as a site projector)", "the spec says 'normalised' without the norm")
    out.flag("collective coupling rate", "sqrt(gamma) P_c taken literally (unit-trace normalisation not applied)", "as written in the spec")
    out.flag("coupling eigenbasis", "local: site basis; collective: the cluster subspaces (degenerate within a cluster), overlap = weight "
             "inside one cluster; random: eigenbasis of sum_m h_m^2", "the h_m do not commute, so random coupling has no common eigenbasis")
    out.flag("prior in sigma", "thermal prior enters as pair weights p_i p_j in sigma; domains are found with the uniform prior",
             "sigma and m_i as defined do not use the prior, so without weights the prior test would be trivially 1")
    out.flag("channel", "Phi^trace only", "Phi^sum needs blocks, which are not defined in rotated bases")

    # checks
    V, blocks = T4.graph("G1", 0.2)
    P = propagator(V.copy(), coupling_ops("local", 8), 1.0, 3.0)
    I8 = np.eye(8, dtype=complex)
    mine = domains_for(in_basis(evolve_basis(P, I8), I8))
    ref = T4.analyse(T4.records(V, np.zeros(8), 1.0, 3.0), blocks, ["trace"])["trace"]
    L.check(set(mine["A"]) == set(ref["A"]) and set(mine["B"]) == set(ref["B"]), "site basis reproduces toy 4")
    res, ov, opt = run_point("G1", 0.2, 0.0, 100.0, 30.0)
    L.check(ov["site"] > 0.9, f"pointer basis approaches the site basis when gamma/V is large ({ov['site']:.3f})")
    out.say(f"Local coupling, gamma = 100, t = 30: pointer overlap with site {ov['site']:.3f}, with exciton {ov['exciton']:.3f}.")
    if not quick:
        res, ov, opt = run_point("G1", 0.2, 0.0, 0.01, 30.0)
        out.say(f"Local coupling, gamma = 0.01, t = 30: pointer overlap with site {ov['site']:.3f}, with exciton {ov['exciton']:.3f} "
                f"(the spec expects the exciton basis here; reported, not asserted, because purity is nearly conserved in every basis as gamma -> 0)")
    out.say("Checks passed: every basis unitary to 1e-10; the site basis reproduces toy 4; the pointer basis approaches the site basis at large gamma.")

    rows_dom, rows_ov, rows_prior, results = [], [], [], {}
    for kind, cseed, p in points(quick):
        res, ov, opt = run_point(*p, kind=kind, cseed=cseed)
        results[(kind, cseed) + p] = res
        rows_ov.append([kind, cseed] + list(p) + [ov["site"], ov["exciton"], ov["coupling"], bool(opt.success), float(opt.fun)])
        for name, r in res.items():
            if name.startswith("random") and name != "random0":
                continue
            for d in ("A", "B"):
                for R in sorted(r[d], key=lambda S: -r["sig_u"][S]):
                    rows_dom.append([kind, cseed] + list(p) + [name, d, T4.fmt_set(R), r["sig_u"][R]])
        for name, r in res.items():
            keys = list(r["sig_u"])
            rho = spearmanr([r["sig_u"][k] for k in keys], [r["sig_t"][k] for k in keys]).statistic if len(keys) >= 3 else float("nan")
            rows_prior.append([kind, cseed] + list(p) + [name, len(keys), rho])
        print(f"{kind} {cseed} {p} done", flush=True)

    # stability: Jaccard between neighbouring t (same gamma) and neighbouring gamma (same t)
    rows_stab = []
    names = ["site", "exciton", "pointer"] + [f"random{k}" for k in range(N_RANDOM_BASES)]
    for (kind, cseed) in sorted({(k, s) for k, s, _ in points(quick)}):
        gl, tl = (RANDOM_G_LIST, RANDOM_T_LIST) if kind == "random" else (G_LIST, T_LIST)
        if quick:
            continue
        for g in ("G1", "G3"):
            for v in V_LIST:
                for W in W_LIST:
                    for name in names:
                        js = []
                        get = lambda ga, t: results[(kind, cseed, g, v, W, ga, t)][name]
                        for ga in gl:
                            for t1, t2 in zip(tl, tl[1:]):
                                a, b = get(ga, t1), get(ga, t2)
                                js.append(jaccard(set(a["A"]) | set(a["B"]), set(b["A"]) | set(b["B"])))
                        for t in tl:
                            for g1, g2 in zip(gl, gl[1:]):
                                a, b = get(g1, t), get(g2, t)
                                js.append(jaccard(set(a["A"]) | set(a["B"]), set(b["A"]) | set(b["B"])))
                        rows_stab.append([kind, cseed, g, v, W, name, float(np.mean(js))])
    for kind in ("local", "collective", "random"):
        sel = [r for r in rows_stab if r[0] == kind]
        if not sel:
            continue
        m = lambda nm: float(np.mean([r[6] for r in sel if r[5] == nm]))
        rnd = float(np.mean([r[6] for r in sel if r[5].startswith("random")]))
        cpl = float(np.mean([r[-3] for r in rows_ov if r[0] == kind]))
        neg = sum(1 for r in rows_prior if r[0] == kind and r[7] == "pointer" and r[9] == r[9] and r[9] < 0)
        out.say(f"{kind} coupling: stability (mean Jaccard) site {m('site'):.3f}, exciton {m('exciton'):.3f}, pointer {m('pointer'):.3f}, "
                f"random bases {rnd:.3f}; pointer overlap with the coupling's own basis {cpl:.3f}; "
                f"points where pointer-basis domains reverse rank under the thermal prior (Spearman < 0): {neg}")

    cols = ["coupling", "coupling seed", "graph", "v", "W", "gamma", "t"]
    hdr = {"event": "an excitation is absorbed by one basis state", "alternatives": "which basis state",
           "prior": "uniform (domains); thermal kT = V (prior test)", "channel": "projection onto span(R) plus outside",
           "random bases": f"{N_RANDOM_BASES}, seed {RANDOM_BASIS_SEED}",
           "random couplings": f"seeds {RANDOM_COUPLING_BASE} + 0..9"}
    out.table("1_domains", hdr, cols + ["basis", "definition", "R", "sigma"], rows_dom,
              note="Random bases: only random0 is listed; all twenty enter the stability and prior tables.")
    out.table("2_stability", hdr, ["coupling", "coupling seed", "graph", "v", "W", "basis", "mean Jaccard (neighbouring t and gamma)"], rows_stab)
    out.table("3_pointer_overlap", hdr, cols + ["overlap with site", "overlap with exciton", "overlap with coupling basis",
                                                "optimiser converged", "summed linear entropy"], rows_ov)
    out.table("4_prior_sensitivity", hdr, cols + ["basis", "domains", "Spearman rho, sigma uniform vs thermal"], rows_prior)
    rows5 = [r[:7] + [r[9]] for r in rows_ov]
    out.table("5_follow_the_coupling", hdr, cols + ["pointer overlap with the coupling's own basis"], rows5,
              note="Pointer-basis domains for each coupling are in 1_domains (basis = pointer).")
    out.finish()


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
