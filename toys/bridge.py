"""The bridge, tested on two toys with one pair of formulas.

Definitions (see BRIDGE.md):
  An event e has alternatives a (which part took it, or which cause produced it),
  with a stated prior p(a). Alternative a leaves a record rho_a in the rest of the
  world. A domain X registers records through a channel Phi_X (its coarse-graining).

  Cut and join (distances), for two alternatives i, j, at X's grain:
      J_ij^X = F(Phi_X rho_i, Phi_X rho_j)          root fidelity (Bhattacharyya for classical)
      D_ij^X = (1/2) || Phi_X rho_i - Phi_X rho_j ||_1   trace distance (total variation)
      Fuchs-van de Graaf:  1 - J <= D <= sqrt(1 - J^2),  hence  J^2 + D^2 <= 1
  Inheritance (information), what e delivers to R:
      E(e -> R) = chi({p_a, Phi_R rho_a})         Holevo information (mutual information if classical)
      data processing: chi can only fall along a chain of channels, so no path sum over-counts
  Conversion, two equiprobable pure records with overlap c (D = sqrt(1 - c^2)):
      chi = h((1 + c)/2)   upper bound; accessible (Helstrom measurement) = 1 - h((1 + D)/2)

Toy 1: a double slit whose which-path record has overlap c, read by X through a
       noisy channel. Includes the three settings of benchmark B10.
Toy 2: a neuron whose soma sums N synapses; the event is one input spike; the
       alternatives are which synapse it came through.

Run: python toys/bridge.py
"""

import math
import warnings

import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import norm

LOG2 = math.log(2)
warnings.filterwarnings("ignore", message="Matrix is singular")


# ---------- quantum records ----------
def fidelity(r, s):
    """Root fidelity F = Tr sqrt(sqrt(r) s sqrt(r))."""
    sr = sqrtm(r)
    return float(np.real(np.trace(sqrtm(sr @ s @ sr))))


def trace_distance(r, s):
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(r - s))))


def entropy_bits(r):
    w = np.linalg.eigvalsh(r)
    w = w[w > 1e-15]
    return float(-np.sum(w * np.log2(w)))


def holevo(priors, states):
    avg = sum(p * s for p, s in zip(priors, states))
    return entropy_bits(avg) - sum(p * entropy_bits(s) for p, s in zip(priors, states))


def h2(x):
    if x <= 0 or x >= 1:
        return 0.0
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def marker_states(c):
    """Two pure marker states with real overlap c."""
    th = math.acos(c)
    a = np.array([1.0, 0.0]); b = np.array([math.cos(th), math.sin(th)])
    return np.outer(a, a), np.outer(b, b)


def depolarize(r, q):
    """X registers the marker only through noise of strength q (0 = perfect, 1 = nothing)."""
    return (1 - q) * r + q * np.eye(2) / 2


def toy1():
    print("TOY 1  Double slit with a which-path marker of overlap c; X reads it through noise q")
    print("   c     q  |  J (join)  D (cut)  J^2+D^2 |  chi bits  accessible bits | visibility seen")
    rows = []
    for c in (0.0, 0.42, 0.9, 0.99):
        for q in (0.0, 0.5, 0.9):
            mL, mR = marker_states(c)
            rL, rR = depolarize(mL, q), depolarize(mR, q)
            J, D = fidelity(rL, rR), trace_distance(rL, rR)
            chi = holevo([0.5, 0.5], [rL, rR])
            acc = 1 - h2((1 + D) / 2)
            rows.append((c, q, J, D, chi, acc))
            print(f"  {c:4.2f}  {q:4.2f} |  {J:6.3f}   {D:6.3f}   {J*J + D*D:6.3f}  |  {chi:6.3f}    {acc:6.3f}"
                  f"        | {c:5.3f}" + ("" if q == 0 else "  (fringes still c: the photon's own record is unchanged)"))
    # equality above is special to pairs read through the same noise; unequal noise makes it strict
    mL, mR = marker_states(0.42)
    rL, rR = depolarize(mL, 0.0), depolarize(mR, 0.7)
    J, D = fidelity(rL, rR), trace_distance(rL, rR)
    print(f"  unequal noise (0 and 0.7) at c = 0.42: J = {J:.3f}, D = {D:.3f}, J^2+D^2 = {J*J + D*D:.3f} (< 1)")
    assert J * J + D * D < 1 - 1e-3 and 1 - J <= D + 1e-9 <= math.sqrt(1 - J * J) + 2e-9
    # check against the pure-state conversion
    for c, q, J, D, chi, acc in rows:
        if q == 0:
            assert abs(J - c) < 1e-6 and abs(chi - h2((1 + c) / 2)) < 1e-6
    print("  B10 settings (the record is the output port; c = V = 2 sqrt(R(1-R))):")
    for R in (0.432, 0.045, 0.0):
        c = 2 * math.sqrt(R * (1 - R))
        D = math.sqrt(1 - c * c)
        print(f"    R = {R:5.3f}: J = {c:.3f}, D = {D:.3f}, chi = {h2((1 + c) / 2):.3f} bits, "
              f"accessible = {1 - h2((1 + D) / 2):.3f} bits")
    return rows


# ---------- classical records ----------
GRID = np.linspace(-15, 25, 8001)   # soma voltage axis, mV


def bhattacharyya(p, q, x=GRID):
    return float(np.trapezoid(np.sqrt(p * q), x))


def total_variation(p, q, x=GRID):
    return float(0.5 * np.trapezoid(np.abs(p - q), x))


def mutual_info_continuous(priors, densities, x=GRID):
    """I(A; V) in bits for a discrete A and a continuous V, densities on the grid."""
    mix = sum(p * d for p, d in zip(priors, densities))
    total = 0.0
    for p, d in zip(priors, densities):
        with np.errstate(divide="ignore", invalid="ignore"):
            integrand = np.where(d > 1e-300, d * np.log2(d / mix), 0.0)
        total += p * np.trapezoid(integrand, x)
    return float(total)


def mutual_info_discrete(priors, cond):
    """I(A; Y) in bits; cond[a] is a probability vector over Y."""
    priors = np.asarray(priors); cond = np.asarray(cond)
    py = priors @ cond
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(cond > 0, cond * np.log2(cond / py), 0.0)
    return float(priors @ terms.sum(axis=1))


def toy2(seed=3):
    """Neuron with N synapses. EPSP sizes lognormal (median 0.5 mV, CV ~ 1); soma noise sigma = 1 mV;
    the output spike fires if the soma voltage exceeds theta = 1.5 mV (toy values, stated)."""
    rng = np.random.default_rng(seed)
    sigma, theta = 1.0, 1.5
    print("\nTOY 2  A neuron summing N synapses; event = one input spike; alternatives = which synapse")
    print("  prior: uniform over the N synapses (stated; another prior gives other numbers)")
    print("     N  | synapse grain | soma grain: mean J  mean D   I(A;V) bits | spike grain: I(A;spike) bits | log2 N")
    out, weights = [], {}
    for N in (2, 10, 100, 1000):
        w = rng.lognormal(mean=math.log(0.5), sigma=0.83, size=N)
        weights[N] = w
        priors = np.full(N, 1 / N)
        dens = [norm.pdf(GRID, wi, sigma) for wi in w]
        # pairwise join and cut at the soma's grain (sample pairs for large N)
        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        if len(pairs) > 2000:
            idx = rng.choice(len(pairs), 2000, replace=False)
            pairs = [pairs[k] for k in idx]
        Js = [math.exp(-(w[i] - w[j]) ** 2 / (8 * sigma**2)) for i, j in pairs]     # Gaussian Bhattacharyya
        Ds = [math.erf(abs(w[i] - w[j]) / (2 * math.sqrt(2) * sigma)) for i, j in pairs]
        I_soma = mutual_info_continuous(priors, dens)
        p_spike = 1 - norm.cdf(theta, w, sigma)
        I_spike = mutual_info_discrete(priors, np.stack([1 - p_spike, p_spike], axis=1))
        out.append((N, np.mean(Js), np.mean(Ds), I_soma, I_spike))
        print(f"  {N:5d} | J=0, D=1, {math.log2(N):5.2f} b |     {np.mean(Js):.3f}    {np.mean(Ds):.3f}     {I_soma:.3f}"
              f"     |           {I_spike:.3f}             | {math.log2(N):.2f}")
        assert I_spike <= I_soma + 1e-6 <= math.log2(N) + 1e-6   # data processing along the chain
    # check the closed-form Bhattacharyya against the grid
    d1, d2 = norm.pdf(GRID, 0.3, sigma), norm.pdf(GRID, 1.7, sigma)
    assert abs(bhattacharyya(d1, d2) - math.exp(-1.4**2 / 8)) < 1e-6
    # redundancy: the same input reaches the soma by two branches with independent noise
    N = 10
    w = weights[N]; pri = np.full(N, 1 / N)
    I1 = mutual_info_continuous(pri, [norm.pdf(GRID, wi, sigma) for wi in w])
    # two independent noisy copies = one copy with noise sigma / sqrt(2)
    I12 = mutual_info_continuous(pri, [norm.pdf(GRID, wi, sigma / math.sqrt(2)) for wi in w])
    print(f"  redundancy, N = 10: one branch {I1:.3f} b; two branches carrying the same input {I12:.3f} b;"
          f" a path sum would claim {2 * I1:.3f} b")
    return out, (I1, I12)


if __name__ == "__main__":
    toy1()
    toy2()
