"""Toy 3: superradiance with mismatched emitters.

N two-level emitters well inside a wavelength share one excitation, starting in the
symmetric state |W>. Collective decay L_c = sqrt(Gamma) sum_k |g><k|; detunings delta_k
~ N(0, Delta^2); local pure dephasing L_k = sqrt(gamma_phi) |k><k|.

In the single-excitation block the master equation is
    d rho/dt = -i[H, rho] - (Gamma/2){A, rho} - gamma_phi (rho - diag rho),   A = all-ones matrix,
and the emitted intensity is I(t) = Gamma * sum_kl rho_kl. Mean emission time and the
enhancement eta = (Gamma_eff/Gamma - 1)/(N - 1) come from exact Laplace moments of the
linear superoperator (no time stepping). One case is checked against QuTiP's mesolve
on the full (N+1)-level space to 1e-6.

Pairwise join between emitters k, l: the record of "emitter k holds the excitation" is the
photon it would emit alone (rate Gamma_rec, detuning delta_k, dephasing gamma_phi), with toy 2's
candidates for mixed records: fidelity F, sqrt Tr(rho_k rho_l), and their squares.
Two versions: Gamma_rec = Gamma and Gamma_rec = N Gamma.

Run: python toys/toy3_superradiance.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, spsolve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge_lib as L  # noqa: E402
from toy2_hom import dephased_matrix  # noqa: E402

N_LIST = [2, 3, 5, 10, 20, 50]
DELTA_LIST = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
GPHI_LIST = [0.0, 0.1, 1.0, 10.0]
N_SEEDS = 50
GAMMA = 1.0
HEADER = {"event": "the shared excitation is emitted", "alternatives": "which emitter k holds the excitation",
          "prior": "uniform over the N emitters", "initial state": "|W> = sum_k |k> / sqrt N"}


def liouvillian(delta, gphi, gamma=GAMMA):
    """Sparse superoperator on vec(rho) (column-major) in the single-excitation block."""
    N = len(delta)
    I = sp.identity(N, format="csr")
    H = sp.diags(delta)
    A = sp.csr_matrix(np.ones((N, N)))
    Lsup = -1j * (sp.kron(I, H) - sp.kron(H.T, I)) - 0.5 * gamma * (sp.kron(I, A) + sp.kron(A.T, I))
    off = np.ones((N, N)) - np.eye(N)
    Lsup = Lsup - gphi * sp.diags(off.flatten(order="F"))
    return Lsup.tocsc()


def initial(N):
    w = np.ones(N) / math.sqrt(N)
    return np.outer(w, w).flatten(order="F").astype(complex)


class FastSolver:
    """Solve L(Y) = -X for the superradiance Liouvillian in O(N^4).

    L(rho) = -i(K rho - rho K^dagger) - gphi rho + gphi diag(rho),  K = H - i Gamma A / 2.
    The part without diag(rho) is a Sylvester map, inverted in K's eigenbasis; the diagonal
    term couples only N unknowns, solved by an N x N linear system.
    """

    def __init__(self, delta, gphi, gamma=GAMMA):
        N = len(delta)
        K = np.diag(delta).astype(complex) - 0.5j * gamma * np.ones((N, N))
        lam, V = np.linalg.eig(K)
        self.V, self.Vi = V, np.linalg.inv(V)
        self.Vh, self.Vih = V.conj().T, self.Vi.conj().T
        self.G = 1.0 / (-1j * (lam[:, None] - lam.conj()[None, :]) - gphi)
        self.gphi, self.N = gphi, N
        if gphi > 0:
            # M[k, l] = diag(S(E_ll))_k
            Bl = np.einsum("il,jl->lij", self.Vi, self.Vi.conj())          # V^-1 E_ll V^-dagger, for each l
            self.M = np.real_if_close(np.array([np.diag(V @ (Bl[l] * self.G) @ self.Vh) for l in range(N)]).T)
            self.lhs = np.eye(N) + gphi * self.M

    def S(self, Z):
        return self.V @ ((self.Vi @ Z @ self.Vih) * self.G) @ self.Vh

    def solve(self, X):
        """Y with L(Y) = -X."""
        Y0 = self.S(-X)
        if self.gphi == 0:
            return Y0
        y = np.linalg.solve(self.lhs, np.diag(Y0))
        return Y0 - self.gphi * self.S(np.diag(y))


def emission_moments(delta, gphi, gamma=GAMMA):
    """Total emitted, <t>, and the fraction emitted by t = 3/(N Gamma)."""
    N = len(delta)
    Lsup = liouvillian(delta, gphi, gamma)
    x0 = initial(N)
    a = np.ones(N * N)                         # I(t) = Gamma * a . vec(rho)
    if np.allclose(delta, delta[0]) and gphi == 0:
        # only the bright state is populated; dark states never mix in: analytic
        total, tmean = 1.0, 1.0 / (N * gamma)
    else:
        fs = FastSolver(delta, gphi, gamma)
        X0 = x0.reshape(N, N, order="F")
        Y1 = fs.solve(X0)                       # integral of rho dt
        Y2 = fs.solve(Y1)                       # integral of t rho dt
        total = float(np.real(gamma * np.sum(Y1)))
        tmean = float(np.real(gamma * np.sum(Y2))) / total
    t3 = 3.0 / (N * gamma)
    xt = expm_multiply(Lsup * t3, x0)
    remaining = float(np.real(np.trace(xt.reshape(N, N, order="F"))))
    early = 1.0 - remaining
    return total, tmean, early, Lsup, x0


def intensity_curve(delta, gphi, times, gamma=GAMMA):
    N = len(delta)
    Lsup = liouvillian(delta, gphi, gamma)
    xs = expm_multiply(Lsup, initial(N), start=times[0], stop=times[-1], num=len(times), endpoint=True)
    I = np.array([gamma * np.real(np.sum(x)) for x in xs])
    pops = np.array([np.real(np.trace(x.reshape(N, N, order="F"))) for x in xs])
    return I, pops


# ---------------------------------------------------------------- pairwise join tables
class JoinTable:
    """Candidates for the join of two dephased exponential photons, as functions of their detuning."""

    def __init__(self, gphi, gamma_rec):
        self.gphi, self.g = gphi, gamma_rec
        self.grid = np.concatenate([[0.0], np.logspace(-3, 3, 61)]) * max(gamma_rec, 1e-12)
        if gphi == 0:
            self.F = None
        else:
            # fidelity of mixed records on a time grid, in units of gamma_rec
            vals = []
            r0, _, _ = dephased_matrix(gphi / gamma_rec, n=260, tmax=13.0)
            for d in self.grid:
                r1, _, _ = dephased_matrix(gphi / gamma_rec, n=260, tmax=13.0, detuning=d / gamma_rec)
                vals.append(float(L.fidelity(r0, r1)))
            self.F = np.array(vals)

    def tr(self, d):
        g, gp = self.g, self.gphi
        return g * (g + 2 * gp) / ((g + 2 * gp) ** 2 + d * d)

    def fid(self, d):
        if self.F is None:                      # pure records: |c| = Gamma / sqrt(Gamma^2 + d^2)
            return self.g / np.sqrt(self.g ** 2 + d * d)
        return np.interp(np.abs(d), self.grid, self.F)


def mean_joins(delta, table):
    N = len(delta)
    i, j = np.triu_indices(N, 1)
    d = delta[i] - delta[j]
    F = table.fid(d); T = table.tr(d)
    return {"F": float(np.mean(F)), "F^2": float(np.mean(F ** 2)),
            "sqrtTr": float(np.mean(np.sqrt(T))), "Tr": float(np.mean(T))}


def collapse_score(rows, key, ykey="eta"):
    """Spread of eta across N at matched join: bin the join into 10 bins and average the
    within-bin standard deviation of eta (lower = better collapse). Also Spearman rho."""
    from scipy.stats import spearmanr
    x = np.array([r[key] for r in rows]); y = np.array([r[ykey] for r in rows])
    bins = np.linspace(0, 1, 11)
    sds = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x >= lo) & (x < hi + (1e-12 if hi == 1 else 0))
        if m.sum() >= 3:
            sds.append(np.std(y[m]))
    return float(np.mean(sds)) if sds else float("nan"), float(spearmanr(x, y).statistic)


def qutip_check(out):
    """One case on the full (N+1)-level space with QuTiP, against the hand-built superoperator."""
    import qutip as qt
    N = 3
    delta = np.array([0.4, -0.2, 0.7]); gphi = 0.3
    dim = N + 1                                   # index 0 = ground, 1..N = excited emitter k
    H = qt.Qobj(np.diag(np.concatenate([[0.0], delta])))
    Lc = sum(qt.basis(dim, 0) * qt.basis(dim, k + 1).dag() for k in range(N)) * math.sqrt(GAMMA)
    Ls = [math.sqrt(gphi) * qt.basis(dim, k + 1) * qt.basis(dim, k + 1).dag() for k in range(N)]
    psi0 = sum(qt.basis(dim, k + 1) for k in range(N)).unit()
    times = np.linspace(0, 4, 41)
    res = qt.mesolve(H, qt.ket2dm(psi0), times, [Lc] + Ls, options={"atol": 1e-12, "rtol": 1e-10})
    I_q = np.array([GAMMA * np.real(np.sum(s.full()[1:, 1:])) for s in res.states])
    I_h, pops = intensity_curve(delta, gphi, times)
    err = float(np.max(np.abs(I_q - I_h)))
    L.check(err < 1e-6, f"QuTiP vs hand-built superoperator: max |dI| = {err}")
    out.say(f"QuTiP mesolve and the hand-built superoperator agree: max |I_qutip - I_hand| = {err:.1e} (N = 3, dephasing 0.3).")


def checks(out):
    # Delta = 0, gamma_phi = 0: I(t) = N Gamma e^{-N Gamma t}
    times = np.linspace(0, 3, 31)
    for N in (1, 2, 5):
        I, pops = intensity_curve(np.zeros(N), 0.0, times)
        L.check(np.max(np.abs(I - N * GAMMA * np.exp(-N * GAMMA * times))) < 1e-9, f"identical emitters, N={N}")
    # N = 2, detunings +-d/2: eigenvalues of H_eff and the intensity they give
    for d in (0.5, 3.0):
        Heff = np.array([[d / 2, 0], [0, -d / 2]]) - 0.5j * GAMMA * np.ones((2, 2))
        ev = np.linalg.eigvals(Heff)
        root = np.sqrt(complex(d * d / 4 - GAMMA ** 2 / 4))
        expected = np.array([-0.5j * GAMMA + root, -0.5j * GAMMA - root])
        err = max(min(abs(e - x) for x in expected) for e in ev)
        L.check(err < 1e-12, f"N = 2 eigenvalues of H_eff: {ev} vs {expected}")
        psi0 = np.ones(2) / math.sqrt(2)
        w, V = np.linalg.eig(Heff)
        c0 = np.linalg.solve(V, psi0)
        I_eig = np.array([GAMMA * abs(np.sum(V @ (c0 * np.exp(-1j * w * t)))) ** 2 for t in times])
        I, _ = intensity_curve(np.array([d / 2, -d / 2]), 0.0, times)
        L.check(np.max(np.abs(I - I_eig)) < 1e-9, f"N = 2 intensity from H_eff eigenvalues, d = {d}")
    # trace: emitted plus remaining = 1
    rng = np.random.default_rng(5)
    delta = rng.normal(0, 1.0, 5)
    Lsup = liouvillian(delta, 0.5)
    x0 = initial(5)
    for t in (0.3, 1.0, 4.0):
        xt = expm_multiply(Lsup * t, x0)
        emitted = float(np.real(GAMMA * np.sum(spsolve(Lsup, xt - x0))))    # exact integral of I from 0 to t
        remaining = float(np.real(np.trace(xt.reshape(5, 5, order="F"))))
        L.check(abs(emitted + remaining - 1) < 1e-10, f"emitted + remaining = 1 at t = {t}: {emitted + remaining}")
    x = expm_multiply(Lsup * 2.0, initial(5)).reshape(5, 5, order="F")
    L.check(np.allclose(x, x.conj().T, atol=1e-12), "rho stays Hermitian")
    rng = np.random.default_rng(9)
    for N, g in ((3, 0.3), (8, 0.0), (10, 10.0)):
        d = rng.normal(0, 1, N)
        x0 = initial(N)
        y = spsolve(-liouvillian(d, g), x0).reshape(N, N, order="F")
        f = FastSolver(d, g).solve(x0.reshape(N, N, order="F"))
        L.check(np.max(np.abs(y - f)) < 1e-10, f"fast solver against sparse solve, N = {N}, gamma_phi = {g}")
    out.say("Checks passed: identical emitters give N Gamma e^{-N Gamma t}; N = 1 gives Gamma e^{-Gamma t}; "
            "N = 2 matches the H_eff eigenvalues; emitted plus remaining population = 1.")


def main():
    out = L.Output("toy3", "Toy 3: superradiance with mismatched emitters")
    out.flag("moments method", "exact Laplace moments: integral of x = (-L)^-1 x0, of t x = (-L)^-2 x0, solved in "
             "O(N^4) via a Sylvester inverse in the eigenbasis of H - i Gamma A/2 plus an N x N correction for dephasing "
             "(checked against a sparse direct solve to 1e-10); "
             "identical emitters without dephasing handled analytically (dark states make L singular)",
             "no time cut-off, so slow subradiant tails are counted in full")
    out.flag("seeds", "seed = 10000*N + 100*index(Delta) + s, s = 0..49; Delta = 0 runs once (no disorder to sample)",
             "fixed and printed, as the spec asks")
    out.flag("pairwise fidelity of mixed records", "260 x 260 time grid to 13/Gamma_rec, tabulated against detuning "
             "and interpolated", "direct fidelity for every pair would take hours")
    out.flag("collective measure eta_early (added)", "eta_early = (f - f_ind)/(f_id - f_ind), f = fraction emitted by t = 3/(N Gamma), "
             "f_ind = 1 - e^{-3/N} (independent emitters), f_id = 1 - e^{-3} (identical)",
             "the spec's eta from the mean emission time is exactly 0 whenever there is any disorder or dephasing (a sum rule: "
             "<t> = 1/Gamma), so it cannot test the join; eta_early is 1 for identical and 0 for independent emitters, like eta")
    out.flag("collapse measure", "mean within-bin standard deviation of eta across all N, join binned in tenths; plus Spearman rho",
             "the spec asks whether one curve collapses the data, with residuals, and forbids fitting a law")
    out.flag("output 3 detector", "spectrometer with bin width b on the emitters' single-photon spectra", "not fixed by the spec")
    checks(out)
    qutip_check(out)

    tables = {}
    rows, rec, sumrule = [], [], []
    for N in N_LIST:
        for gp in GPHI_LIST:
            for gr_name, gr in (("Gamma", GAMMA), ("N Gamma", N * GAMMA)):
                key = (gp, round(gr, 6))
                if key not in tables:
                    tables[key] = JoinTable(gp, gr)
            for di, Dl in enumerate(DELTA_LIST):
                seeds = [0] if Dl == 0 else range(N_SEEDS)
                etas, earlies, joins = [], [], {"Gamma": [], "N Gamma": []}
                for s in seeds:
                    rng = np.random.default_rng(10000 * N + 100 * di + s)
                    delta = rng.normal(0, Dl, N) if Dl > 0 else np.zeros(N)
                    total, tmean, early, _, _ = emission_moments(delta, gp)
                    eta = (1 / (tmean * GAMMA) - 1) / (N - 1)
                    f_ind, f_id = 1 - math.exp(-3.0 / N), 1 - math.exp(-3.0)
                    eta_e = (early - f_ind) / (f_id - f_ind)
                    if not (Dl == 0 and gp == 0):
                        sumrule.append(abs(tmean * GAMMA - 1))
                    etas.append(eta); earlies.append(eta_e)
                    jr = {}
                    for gr_name, gr in (("Gamma", GAMMA), ("N Gamma", N * GAMMA)):
                        j = mean_joins(delta, tables[(gp, round(gr, 6))])
                        joins[gr_name].append(j)
                        for cand in ("F", "F^2", "sqrtTr", "Tr"):
                            jr[f"{cand} ({gr_name})"] = j[cand]
                    rec.append({"N": N, "Delta": Dl, "gphi": gp, "seed": s, "eta": eta, "eta_early": eta_e, "early": early, **jr})
                mj = {f"{c} ({g})": float(np.mean([j[c] for j in joins[g]])) for g in joins for c in ("F", "sqrtTr")}
                rows.append([N, Dl, gp, len(list(seeds)), float(np.mean(etas)), float(np.std(etas)),
                             float(np.mean(earlies)), float(np.std(earlies)),
                             mj["F (Gamma)"], mj["sqrtTr (Gamma)"], mj["F (N Gamma)"], mj["sqrtTr (N Gamma)"]])
        print(f"N = {N} done", flush=True)
    out.table("1_eta", {**HEADER, "channel": "the emitted field (J_phys); joins from single-emitter photon records",
                        "seeds": "50 per disorder point, seed = 10000 N + 100 i_Delta + s"},
              ["N", "Delta/Gamma", "gamma_phi/Gamma", "seeds", "eta (from <t>) mean", "eta sd", "eta_early mean", "eta_early sd",
               "mean F (Gamma)", "mean sqrtTr (Gamma)", "mean F (N Gamma)", "mean sqrtTr (N Gamma)"], rows)

    # output 2: the test
    cands = [f"{c} ({g})" for g in ("Gamma", "N Gamma") for c in ("F", "F^2", "sqrtTr", "Tr")]
    import csv
    with open(out.dir / "2_eta_vs_join.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"# event: {HEADER['event']}; prior uniform; one row per disorder draw"])
        w.writerow(["N", "Delta/Gamma", "gamma_phi/Gamma", "seed", "eta (from <t>)", "eta_early", "early fraction"] + cands)
        for r in rec:
            w.writerow([r["N"], r["Delta"], r["gphi"], r["seed"], f"{r['eta']:.8g}", f"{r['eta_early']:.8g}", f"{r['early']:.8g}"] + [f"{r[c]:.8g}" for c in cands])
    multi = [r for r in rec if r["N"] > 2]
    score_rows = []
    for c in cands:
        sd, rho = collapse_score(multi, c, "eta")
        sde, rhoe = collapse_score(multi, c, "eta_early")
        score_rows.append([c, sd, rho, sde, rhoe])
    out.table("2_collapse", {**HEADER, "data": "every disorder draw with N >= 3 (N = 2 has a single pair)",
                             "measure": "within-bin sd across N at matched join (lower = tighter collapse); Spearman rho"},
              ["join candidate (record timescale)", "sd of eta (from <t>)", "Spearman (eta, join)",
               "sd of eta_early", "Spearman (eta_early, join)"], score_rows,
              note="Per-draw data are in 2_eta_vs_join.csv. No law is fitted. eta from <t> is 0 for every draw with any "
                   "disorder or dephasing (see SUMMARY), so its scores carry no information.")
    out.say(f"Sum rule: with any disorder or dephasing, the mean emission time <t> = 1/Gamma exactly "
            f"(largest deviation {max(sumrule):.1e} over {len(sumrule)} runs), so eta from <t> is 0 except for "
            f"identical emitters. Slowly decaying trapped population exactly offsets the early burst in the mean.")
    for c, sd, rho, sde, rhoe in score_rows:
        out.say(f"Collapse, {c}: eta_early within-bin sd = {sde:.3f}, Spearman rho = {rhoe:.3f}")
    best = min(score_rows, key=lambda r: r[3])
    out.say(f"Tightest collapse of eta_early: {best[0]}")

    # output 3: physical join fixed, detector resolution varies
    rows = []
    rng = np.random.default_rng(77)
    N, Dl, gp = 10, 1.0, 0.0
    delta = rng.normal(0, Dl, N)
    total, tmean, early, _, _ = emission_moments(delta, gp)
    eta = (1 / tmean - 1) / (N - 1)
    eta_e3 = (early - (1 - math.exp(-3.0 / N))) / ((1 - math.exp(-3.0)) - (1 - math.exp(-3.0 / N)))
    for bw in np.logspace(-2, 2, 9):
        edges = np.arange(-60, 60 + bw, bw)
        cdf = lambda mu: 0.5 + np.arctan(2 * (edges - mu)) / math.pi     # Lorentzian, FWHM Gamma
        specs = [np.diff(cdf(d)) for d in delta]
        specs = [s / s.sum() for s in specs]
        i, j = np.triu_indices(N, 1)
        Jx = float(np.mean([L.bhattacharyya(specs[a], specs[b]) for a, b in zip(i, j)]))
        rows.append([bw, Jx, mean_joins(delta, tables[(0.0, 1.0)])["F"], eta, eta_e3])
    out.table("3_detector_resolution", {**HEADER, "N": 10, "Delta/Gamma": 1.0, "gamma_phi": 0, "seed": 77,
                                        "channel": "spectrometer bins of width b on each emitter's photon spectrum"},
              ["bin width / Gamma", "J^X (mean over pairs)", "J_phys = mean F (Gamma)", "eta (from <t>)", "eta_early"], rows,
              note="eta is a property of the emitted field and does not change with the detector.")
    out.say(f"Detector resolution from 0.01 to 100 Gamma moves the registered join from {rows[0][1]:.3f} to {rows[-1][1]:.3f}; "
            f"eta_early stays {eta_e3:.4f} (eta from <t> stays {eta:.4f}).")
    out.finish()


if __name__ == "__main__":
    main()
