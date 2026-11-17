"""Shared library for the bridge toys (spec: Bridge toys specification, 16 Nov 2026).

Measures
  join   J = root fidelity F(rho, sigma) = Tr|sqrt(rho) sqrt(sigma)|   (Bhattacharyya if classical)
  cut    D = trace distance (1/2)||rho - sigma||_1                    (total variation if classical)
  inheritance  chi = Holevo quantity, in bits                          (mutual information if classical)
Channels: partial trace, depolarizing, binning, projection onto a subspace plus "outside".
Domain measures: m_i(R), sigma(R), fixed-point definitions A (local maximum) and B (self-consistency).
Checks raise CheckFailed: they fail loudly and never only warn.
Output: Markdown and CSV tables with a header block; flagged modelling choices.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

TOYS = Path(__file__).resolve().parent
RESULTS = TOYS / "results"
EPS = 1e-12


class CheckFailed(AssertionError):
    pass


def check(condition, message):
    if not bool(condition):
        raise CheckFailed(message)


# ---------------------------------------------------------------- basic functions
def h2(x):
    """Binary entropy in bits."""
    x = float(x)
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def shannon(p):
    p = np.asarray(p, float)
    p = p[p > EPS]
    return float(-np.sum(p * np.log2(p)))


def psd_sqrt(rho):
    w, v = np.linalg.eigh((rho + rho.conj().swapaxes(-1, -2)) / 2)
    w = np.clip(w, 0, None)
    return (v * np.sqrt(w)[..., None, :]) @ v.conj().swapaxes(-1, -2)


def fidelity(rho, sigma):
    """Root fidelity F = Tr sqrt(sqrt(rho) sigma sqrt(rho)). Works on stacks (..., d, d)."""
    s = psd_sqrt(rho)
    m = s @ sigma @ s
    w = np.linalg.eigvalsh((m + m.conj().swapaxes(-1, -2)) / 2)
    return np.sum(np.sqrt(np.clip(w, 0, None)), axis=-1)


def trace_distance(rho, sigma):
    w = np.linalg.eigvalsh(rho - sigma)
    return 0.5 * np.sum(np.abs(w), axis=-1)


def vn_entropy(rho):
    w = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    return shannon(w)


def holevo(priors, states):
    priors = np.asarray(priors, float)
    avg = sum(p * s for p, s in zip(priors, states))
    return vn_entropy(avg) - float(sum(p * vn_entropy(s) for p, s in zip(priors, states)))


def accessible_two_pure(c):
    """Accessible information (bits) of two equiprobable pure states with overlap |c|: 1 - h((1+D)/2)."""
    D = math.sqrt(max(0.0, 1 - c * c))
    return 1 - h2((1 + D) / 2)


def purity(rho):
    return float(np.real(np.trace(rho @ rho)))


def overlap_trace(rho, sigma):
    """Tr(rho sigma), what HOM interference measures."""
    return float(np.real(np.trace(rho @ sigma)))


# ---------------------------------------------------------------- classical records
def bhattacharyya(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(np.sum(np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None))))


def total_variation(p, q):
    return float(0.5 * np.sum(np.abs(np.asarray(p, float) - np.asarray(q, float))))


def mutual_information(priors, cond):
    """I(A; Y) in bits; cond[a] is a probability vector over Y."""
    priors = np.asarray(priors, float)
    cond = np.asarray(cond, float)
    py = priors @ cond
    return shannon(py) - float(sum(p * shannon(c) for p, c in zip(priors, cond)))


# ---------------------------------------------------------------- channels
def ket(*amps):
    v = np.array(amps, dtype=complex)
    return v / np.linalg.norm(v)


def dm(v):
    v = np.asarray(v, dtype=complex)
    return np.outer(v, v.conj())


def partial_trace(rho, dims, keep):
    """Keep the subsystems listed in keep (indices into dims); trace out the rest."""
    dims = list(dims)
    n = len(dims)
    keep = sorted(keep)
    rho = rho.reshape(dims + dims)
    trace_out = [i for i in range(n) if i not in keep]
    # trace pairs from the highest index down so axis numbers stay valid
    cur_n = n
    for i in sorted(trace_out, reverse=True):
        rho = np.trace(rho, axis1=i, axis2=i + cur_n)
        cur_n -= 1
    d = int(np.prod([dims[i] for i in keep])) if keep else 1
    return rho.reshape(d, d)


def depolarize(rho, q):
    """(1 - q) rho + q I/d."""
    d = rho.shape[-1]
    return (1 - q) * rho + q * np.eye(d) / d


def bin_distribution(values, weights, edges):
    """Classical binning channel: put weighted values into bins; returns probabilities."""
    hist, _ = np.histogram(values, bins=edges, weights=weights)
    s = hist.sum()
    return hist / s if s > 0 else hist


def project_with_outside(rho, members):
    """Phi^trace for a set R of basis states: the block of rho on R, plus one 'outside' state
    holding the population 1 - Tr(rho_R); no coherence between the two."""
    idx = list(members)
    block = rho[np.ix_(idx, idx)]
    out = max(0.0, 1.0 - float(np.real(np.trace(block))))
    d = len(idx) + 1
    res = np.zeros((d, d), dtype=complex)
    res[:-1, :-1] = block
    res[-1, -1] = out
    return res


# ---------------------------------------------------------------- domain measures
def pair_matrix(records, channel):
    """J and D matrices over all alternatives, at a channel's grain. records: list of states."""
    n = len(records)
    reg = [channel(r) for r in records]
    J = np.ones((n, n))
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = J[j, i] = float(fidelity(reg[i], reg[j]))
            D[i, j] = D[j, i] = float(trace_distance(reg[i], reg[j]))
    return J, D


def membership(J, R):
    """m_i(R) for every alternative i, using J at R's grain."""
    R = list(R)
    n = J.shape[0]
    m = np.zeros(n)
    for i in range(n):
        others = [j for j in R if j != i]
        m[i] = J[i, others].mean() if others else 0.0
    return m


def sigma_of(J, R, weights=None):
    """sigma(R) = mean J over pairs inside R minus mean J over pairs across R's edge.
    weights: optional prior; pairs are then weighted by p_i p_j (used only for prior tests)."""
    R = set(R)
    n = J.shape[0]
    inside, across, wi, wa = 0.0, 0.0, 0.0, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            w = 1.0 if weights is None else weights[i] * weights[j]
            if i in R and j in R:
                inside += w * J[i, j]; wi += w
            elif (i in R) != (j in R):
                across += w * J[i, j]; wa += w
    if wi == 0 or wa == 0:
        return float("nan")
    return inside / wi - across / wa


def is_domain_A(sigma_fn, R, n):
    """Definition A: adding or removing any single part lowers sigma(R)."""
    R = frozenset(R)
    s = sigma_fn(R)
    if not np.isfinite(s):
        return False
    for k in range(n):
        R2 = R - {k} if k in R else R | {k}
        if 2 <= len(R2) <= n - 1:
            s2 = sigma_fn(R2)
            if np.isfinite(s2) and s2 >= s:
                return False
    return True


def is_domain_B(J_R, R):
    """Definition B: every member's m_i(R) exceeds every non-member's, both at R's grain."""
    m = membership(J_R, R)
    inside = [m[i] for i in R]
    outside = [m[j] for j in range(J_R.shape[0]) if j not in R]
    if not outside:
        return False
    return min(inside) > max(outside)


def subsets(n, min_size=2, max_size=None):
    max_size = n - 1 if max_size is None else max_size
    for k in range(min_size, max_size + 1):
        for R in itertools.combinations(range(n), k):
            yield frozenset(R)


# ---------------------------------------------------------------- run-wide checks
def check_pair(J, D, pure=False, tol=1e-9, label=""):
    check(1 - J <= D + tol, f"Fuchs-van de Graaf lower bound broken {label}: J={J}, D={D}")
    check(D <= math.sqrt(max(0.0, 1 - J * J)) + tol, f"Fuchs-van de Graaf upper bound broken {label}: J={J}, D={D}")
    if pure:
        check(abs(J * J + D * D - 1) <= 1e-10, f"pure records: J^2 + D^2 != 1 {label}: {J*J + D*D}")


def check_monotone(J_coarse, J_fine, tol=1e-9, label=""):
    check(J_coarse >= J_fine - tol, f"monotonicity broken {label}: coarse {J_coarse} < fine {J_fine}")


def check_dpi(chain, H_prior, tol=1e-9, label=""):
    """chain: information values along a chain of channels, finest first."""
    for a, b in zip(chain, chain[1:]):
        check(b <= a + tol, f"data processing broken {label}: {chain}")
    check(chain[0] <= H_prior + tol, f"information exceeds H(prior) {label}: {chain[0]} > {H_prior}")


SATURATION_NOTE = ("Saturating J^2 + D^2 = 1 does not show records are quantum: two mirror-image "
                   "binary distributions also saturate it.")


# ---------------------------------------------------------------- output
class Output:
    """Collects tables and flagged choices for one toy and writes them to toys/results/toyN/."""

    def __init__(self, toy: str, title: str):
        self.toy, self.title = toy, title
        self.dir = RESULTS / toy
        self.dir.mkdir(parents=True, exist_ok=True)
        self.flags = []
        self.summary = []

    def flag(self, choice, value, why):
        self.flags.append({"toy": self.toy, "choice": choice, "value": str(value), "why": why})

    def table(self, name, header: dict, columns, rows, note=""):
        """Write name.md and name.csv. header: event, alternatives, prior, channel, time, seed..."""
        rows = [list(r) for r in rows]
        with open(self.dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for k, v in header.items():
                w.writerow([f"# {k}: {v}"])
            w.writerow(columns)
            for r in rows:
                w.writerow([_csvfmt(x) for x in r])
        lines = [f"# {self.title}: {name}", ""]
        lines += [f"- **{k}:** {v}" for k, v in header.items()]
        if note:
            lines += ["", note]
        lines += ["", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        lines += ["| " + " | ".join(_mdfmt(x) for x in r) + " |" for r in rows]
        (self.dir / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def say(self, line):
        print(line)
        self.summary.append(line)

    def finish(self):
        (self.dir / "flags.json").write_text(json.dumps(self.flags, indent=2) + "\n", encoding="utf-8")
        text = [f"# {self.title}: summary", "", "Key numbers only, with no interpretation.", ""]
        text += [s if s.startswith(("|", "#", "-")) or s == "" else f"- {s}" for s in self.summary]
        (self.dir / "SUMMARY.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def _csvfmt(x):
    if isinstance(x, float):
        return f"{x:.10g}"
    return x


def _mdfmt(x):
    if isinstance(x, (float, np.floating)):
        if x != x:
            return "–"
        if x == 0:
            return "0"
        if abs(x) >= 1e4 or abs(x) < 1e-3:
            return f"{x:.3e}"
        return f"{x:.4g}"
    return str(x)


def build_flags_md():
    """Collect every toy's flags.json into toys/FLAGS.md."""
    rows = []
    for f in sorted(RESULTS.glob("toy*/flags.json")):
        rows += json.loads(f.read_text(encoding="utf-8"))
    lines = ["# Flagged choices", "",
             "Every modelling choice the spec did not fix, with the value used and why. "
             "Generated from toys/results/toyN/flags.json by `python toys/run_toys.py --flags`.", "",
             "| Toy | Choice | Value used | Why |", "|---|---|---|---|"]
    lines += [f"| {r['toy']} | {r['choice']} | {r['value']} | {r['why']} |" for r in rows]
    (TOYS / "FLAGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(rows)
