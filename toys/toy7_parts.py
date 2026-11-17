"""Toy 7: whose parts? Does the pointer basis fix the choice of parts?

Toy 4's graphs G1 and G3 (same H, dephasing gamma, disorder W, time t). The alternatives are
"which basis state took the excitation"; a candidate domain R is a set of basis states,
registered by projection onto their span plus 'outside' (toy 4's Phi^trace, in the new basis).

Bases: site (must reproduce toy 4), exciton (eigenstates of H), pointer (the orthonormal basis
whose members, each evolved alone to time t, lose least purity: the predictability sieve),
and 20 Haar-random bases (the null). Priors: uniform, and thermal p_a ~ exp(-<a|H|a>/kT), kT = V.

Counts against the rule: pointer-basis domains no more stable than random-basis domains, or
domains that reverse rank when the prior changes.

Run: python toys/toy7_parts.py           (reduced sweep, see FLAGS)
     python toys/toy7_parts.py --quick   (checks and one point)
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

N_RANDOM = 20
RANDOM_SEED = 7000
V_LIST = [0.05, 0.2]
W_LIST = [0.0, 0.5]
G_LIST = T4.G_LIST
T_LIST = T4.T_LIST
KT = 1.0


def haar(n, rng):
    Z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / math.sqrt(2)
    Q, R = np.linalg.qr(Z)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def propagator(H, gamma, t):
    return expm(T4.liouvillian(H, gamma) * t)


def evolve_basis(P, U):
    """Each basis vector (column of U), evolved alone; returned as density matrices in the site basis."""
    n = U.shape[0]
    out = []
    for a in range(n):
        r0 = np.outer(U[:, a], U[:, a].conj())
        out.append((P @ r0.flatten(order="F")).reshape(n, n, order="F"))
    return np.array(out)


def pointer_basis(P, n, start, iters=300):
    """Minimise summed linear entropy 1 - Tr rho_a(t)^2 over unitaries U = start expm(i X), X Hermitian."""
    def unitary(x):
        X = np.zeros((n, n), complex)
        iu = np.triu_indices(n, 1)
        X[iu] = x[: len(iu[0])] + 1j * x[len(iu[0]): 2 * len(iu[0])]
        X = X + X.conj().T
        X[np.diag_indices(n)] = x[2 * len(iu[0]):]
        return start @ expm(1j * X)

    def cost(x):
        recs = evolve_basis(P, unitary(x))
        return float(np.sum(1 - np.real(np.einsum("aij,aji->a", recs, recs))))

    x0 = np.zeros(n * n)
    res = minimize(cost, x0, method="L-BFGS-B", options={"maxiter": iters})
    return unitary(res.x), res


def in_basis(recs_site, U):
    return np.einsum("ji,ajk,kl->ail", U.conj(), recs_site, U)


def domains_for(recs_basis):
    res = T4.analyse(recs_basis, None, ["trace"])["trace"]
    return res


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def run_point(gname, v, W, gamma, t, rng_seed=RANDOM_SEED):
    V, _ = T4.graph(gname, v)
    n = V.shape[0]
    H = np.diag(T4.disorder(n, W)) + V
    P = propagator(H, gamma, t)
    _, Uex = np.linalg.eigh(H)
    # pointer: start from the better of site and exciton bases
    c_site = np.sum(1 - np.real(np.einsum("aij,aji->a", *(2 * [evolve_basis(P, np.eye(n))]))))
    c_ex = np.sum(1 - np.real(np.einsum("aij,aji->a", *(2 * [evolve_basis(P, Uex)]))))
    start = np.eye(n, dtype=complex) if c_site <= c_ex else Uex.astype(complex)
    Up, opt = pointer_basis(P, n, start)
    rng = np.random.default_rng(rng_seed)
    bases = {"site": np.eye(n, dtype=complex), "exciton": Uex.astype(complex), "pointer": Up}
    for k in range(N_RANDOM):
        bases[f"random{k}"] = haar(n, rng)
    out = {}
    for name, U in bases.items():
        L.check(np.max(np.abs(U.conj().T @ U - np.eye(n))) < 1e-10, f"basis {name} unitary")
        recs = in_basis(evolve_basis(P, U), U)
        res = domains_for(recs)
        energies = np.real(np.einsum("ia,ij,ja->a", U.conj(), H, U))
        prior = np.exp(-(energies - energies.min()) / KT); prior /= prior.sum()
        dom = set(res["A"]) | set(res["B"])
        sig_u = {R: res["sigma"][R] for R in dom}
        sig_t = {R: L.sigma_of(res["J"][R], R, weights=prior) for R in dom}
        out[name] = {"A": res["A"], "B": res["B"], "sig_u": sig_u, "sig_t": sig_t, "U": U}
    overlaps = {nm: float(np.mean(np.max(np.abs(bases[nm].conj().T @ Up) ** 2, axis=0))) for nm in ("site", "exciton")}
    return out, overlaps, opt


def main(quick=False):
    out = L.Output("toy7", "Toy 7: whose parts?")
    out.flag("reduced sweep", f"G1 and G3; v in {V_LIST}; W in {W_LIST}; all gamma and t",
             "the full toy-4 sweep times 23 bases would take hours")
    out.flag("pointer optimiser", "L-BFGS-B on the Hermitian generator of U = U0 expm(iX), started from the better of site and "
             "exciton bases, 300 iterations; convergence status printed per point", "the spec asks to flag the optimiser")
    out.flag("prior in sigma", "thermal prior enters as pair weights p_i p_j in sigma; domains are found with the uniform prior",
             "sigma and m_i as defined do not use the prior, so without weights the prior test would be trivially 1")
    out.flag("thermal prior energies", "<a|H|a> in each basis, shifted by the minimum; kT = V = 1", "as the spec states")
    out.flag("channel", "Phi^trace only (projection onto the span of R's basis states plus outside)",
             "Phi^sum needs blocks, which are not defined in rotated bases")

    # checks: site basis reproduces toy 4
    V, blocks = T4.graph("G1", 0.2)
    H = V.copy()
    P = propagator(H, 1.0, 3.0)
    recs = in_basis(evolve_basis(P, np.eye(8, dtype=complex)), np.eye(8, dtype=complex))
    mine = domains_for(recs)
    ref = T4.analyse(T4.records(V, np.zeros(8), 1.0, 3.0), blocks, ["trace"])["trace"]
    L.check(set(mine["A"]) == set(ref["A"]) and set(mine["B"]) == set(ref["B"]), "site basis reproduces toy 4")
    # limits of the pointer basis at long t
    for gamma, target in ((100.0, "site"), (0.01, "exciton")):
        res, ov, opt = run_point("G1", 0.2, 0.0, gamma, 30.0) if not quick or gamma == 100.0 else (None, None, None)
        if ov is None:
            continue
        out.say(f"Pointer basis at gamma = {gamma}, t = 30: mean overlap with site {ov['site']:.3f}, with exciton {ov['exciton']:.3f} "
                f"(expected closer to the {target} basis); optimiser success = {opt.success}")
        if gamma == 100.0:
            L.check(ov["site"] > 0.9, "pointer basis approaches the site basis when gamma/V is large")
    out.say("Checks passed: every basis unitary to 1e-10; the site basis reproduces toy 4.")

    pts = [("G1", 0.2, 0.0, 1.0, 3.0)] if quick else [(g, v, W, ga, t) for g in ("G1", "G3") for v in V_LIST for W in W_LIST for ga in G_LIST for t in T_LIST]
    results = {}
    rows_dom, rows_ov, rows_prior = [], [], []
    for p in pts:
        res, ov, opt = run_point(*p)
        results[p] = res
        rows_ov.append(list(p) + [ov["site"], ov["exciton"], bool(opt.success), float(opt.fun)])
        for name, r in res.items():
            if name.startswith("random") and name != "random0":
                continue
            for d in ("A", "B"):
                for R in sorted(r[d], key=lambda S: -r["sig_u"][S]):
                    rows_dom.append(list(p) + [name, d, T4.fmt_set(R), r["sig_u"][R]])
        for name, r in res.items():
            keys = list(r["sig_u"])
            if len(keys) >= 3:
                rho = spearmanr([r["sig_u"][k] for k in keys], [r["sig_t"][k] for k in keys]).statistic
            else:
                rho = float("nan")
            rows_prior.append(list(p) + [name, len(keys), rho])
        print(f"point {p} done", flush=True)

    # stability: Jaccard between neighbouring t (same gamma) and neighbouring gamma (same t)
    rows_stab = []
    if not quick:
        for g in ("G1", "G3"):
            for v in V_LIST:
                for W in W_LIST:
                    for name in ["site", "exciton", "pointer"] + [f"random{k}" for k in range(N_RANDOM)]:
                        js = []
                        for ga in G_LIST:
                            for t1, t2 in zip(T_LIST, T_LIST[1:]):
                                a = results[(g, v, W, ga, t1)][name]; b = results[(g, v, W, ga, t2)][name]
                                js.append(jaccard(set(a["A"]) | set(a["B"]), set(b["A"]) | set(b["B"])))
                        for t in T_LIST:
                            for g1, g2 in zip(G_LIST, G_LIST[1:]):
                                a = results[(g, v, W, g1, t)][name]; b = results[(g, v, W, g2, t)][name]
                                js.append(jaccard(set(a["A"]) | set(a["B"]), set(b["A"]) | set(b["B"])))
                        rows_stab.append([g, v, W, name, float(np.mean(js))])
        for g in ("G1", "G3"):
            sel = [r for r in rows_stab if r[0] == g]
            mean = lambda nm: float(np.mean([r[4] for r in sel if r[3] == nm]))
            rnd = float(np.mean([r[4] for r in sel if r[3].startswith("random")]))
            out.say(f"{g} stability (mean Jaccard): site {mean('site'):.3f}, exciton {mean('exciton'):.3f}, "
                    f"pointer {mean('pointer'):.3f}, random {rnd:.3f}")
    cols = ["graph", "v", "W", "gamma", "t"]
    hdr = {"event": "an excitation is absorbed by one basis state", "alternatives": "which basis state",
           "prior": "uniform (domains); thermal kT = V (prior test)", "channel": "projection onto span(R) plus outside",
           "random bases": f"{N_RANDOM}, seed {RANDOM_SEED}"}
    out.table("1_domains", hdr, cols + ["basis", "definition", "R", "sigma"], rows_dom,
              note="Random bases: only random0 is listed here; all twenty enter the stability and prior tables.")
    out.table("2_stability", hdr, ["graph", "v", "W", "basis", "mean Jaccard (neighbouring t and gamma)"], rows_stab)
    out.table("3_pointer_overlap", hdr, cols + ["mean overlap with site", "mean overlap with exciton", "optimiser converged", "summed linear entropy"], rows_ov)
    out.table("4_prior_sensitivity", hdr, cols + ["basis", "domains", "Spearman rho, sigma uniform vs thermal"], rows_prior)
    out.finish()


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
