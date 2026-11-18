"""Toy 1b: a chain of events (the reach rule).

A qubit S in (|0> + |1>)/sqrt2 (alternatives: path 0, path 1) passes through events E1..En, then reaches
an end event: a beam splitter of reflectivity R, then detection. Each event writes record qubits by a
controlled operation. The partial order is an explicit directed acyclic graph; P(e) is every event
before e in it.

Each path's records are simulated as a state vector over all record qubits (S's path is the branch
label), so every quantity comes from the states themselves:
  J_e   fidelity of all records in P(e), plus e's own outputs, taken jointly
  J_phys fidelity of every record anywhere (including events outside P(e))
  J_out fidelity of e's own outputs alone (end event: Bhattacharyya of the port distributions; flagged)
  V     fringe visibility from scanning the phase and computing port probabilities; V_max = 2 sqrt(R(1-R))

Run: python toys/toy1b_chain.py
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge_lib as L  # noqa: E402

TOL = 1e-10
HEADER = {"event": "the end event: beam splitter of reflectivity R, then detection",
          "alternatives": "path 0 or path 1", "prior": "uniform (1/2, 1/2)"}


# ---------------------------------------------------------------- a small register of record qubits
class Chain:
    """Record qubits written by events; each path a has its own state vector over them."""

    def __init__(self, name):
        self.name = name
        self.events = []          # (name, [qubit indices], parents)
        self.n = 0
        self.ops = []             # list of (event index, function(vec_by_path) -> vec_by_path)

    def add_qubits(self, k):
        idx = list(range(self.n, self.n + k))
        self.n += k
        return idx

    def event(self, name, outputs, parents, writer):
        """writer(path) -> state vector over the event's output qubits (fresh qubits), or a
        two-path unitary acting on existing qubits when outputs is given as (existing, fresh)."""
        self.events.append((name, outputs, parents))
        self.ops.append((len(self.events) - 1, writer))

    def branches(self, upto=None):
        """State vector per path over all qubits written by events (in order) up to index upto."""
        vecs = {0: np.array([1.0 + 0j]), 1: np.array([1.0 + 0j])}
        dims = 0
        for ei, writer in self.ops[: (len(self.ops) if upto is None else upto + 1)]:
            kind, payload = writer
            if kind == "fresh":                       # new qubits in a path-dependent state
                for a in (0, 1):
                    vecs[a] = np.kron(vecs[a], payload(a))
                dims += int(round(math.log2(len(payload(0)))))
            elif kind == "measure":                   # copy an existing qubit into a fresh meter in a basis
                q, basis = payload
                for a in (0, 1):
                    vecs[a] = measure_copy(vecs[a], dims, q, basis)
                dims += 1
        return vecs, dims


def measure_copy(vec, nq, q, basis):
    """A measuring event: copy qubit q (in basis 'Z' or 'X') into a fresh meter qubit appended at the end."""
    psi = vec.reshape([2] * nq)
    if basis == "X":
        Hd = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
        psi = np.moveaxis(np.tensordot(Hd, psi, axes=([1], [q])), 0, q)
    out = np.zeros([2] * nq + [2], complex)
    sl0 = [slice(None)] * nq; sl0[q] = 0
    sl1 = [slice(None)] * nq; sl1[q] = 1
    out[tuple(sl0) + (0,)] = psi[tuple(sl0)]
    out[tuple(sl1) + (1,)] = psi[tuple(sl1)]
    if basis == "X":
        Hd = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
        out = np.moveaxis(np.tensordot(Hd, out, axes=([1], [q])), 0, q)
    return out.reshape(-1)


def reduced(vec, nq, keep):
    if not keep:
        return np.array([[1.0 + 0j]])
    return L.partial_trace(np.outer(vec, vec.conj()), [2] * nq, keep)


def fid_over(vecs, nq, keep):
    if not keep:
        return 1.0
    return float(L.fidelity(reduced(vecs[0], nq, keep), reduced(vecs[1], nq, keep)))


def marker(c):
    return lambda a: np.array([1.0, 0.0], complex) if a == 0 else np.array([c, math.sqrt(max(0.0, 1 - c * c))], complex)


def same(state):
    s = np.asarray(state, complex)
    return lambda a: s / np.linalg.norm(s)


def fringes(vecs, nq, R, project=None):
    """Visibility of port-0 probability against phase, from S's (conditional) state and the splitter."""
    v0, v1 = vecs[0], vecs[1]
    if project is not None:
        v0, v1 = project(v0), project(v1)
    rho_S = 0.5 * np.array([[np.vdot(v0, v0), np.vdot(v1, v0)], [np.vdot(v0, v1), np.vdot(v1, v1)]])
    norm = np.real(np.trace(rho_S))
    if norm < 1e-15:
        return float("nan"), float("nan"), 0.0
    rho_S = rho_S / norm
    r, t = math.sqrt(R), math.sqrt(1 - R)
    U = np.array([[t, 1j * r], [1j * r, t]])
    phis = np.linspace(0, 2 * math.pi, 721)
    P0 = []
    for ph in phis:
        Ph = np.diag([1, np.exp(1j * ph)])
        M = U @ Ph @ rho_S @ Ph.conj().T @ U.conj().T
        P0.append(np.real(M[0, 0]))
    P0 = np.array(P0)
    V = (P0.max() - P0.min()) / (P0.max() + P0.min())
    return float(V), float(phis[int(np.argmax(P0))]), float(norm)


def port_bhattacharyya(R):
    return L.bhattacharyya([1 - R, R], [R, 1 - R])


# ---------------------------------------------------------------- the cases
def build(case, **kw):
    ch = Chain(case)
    if case == "A":
        for k in range(3):
            ch.event(f"E{k+1}", ch.add_qubits(1), [f"E{k}"] if k else [], ("fresh", same([0.6, 0.8])))
    elif case == "B":
        for k, c in enumerate(kw["cs"]):
            ch.event(f"E{k+1}", ch.add_qubits(1), [f"E{k}"] if k else [], ("fresh", marker(c)))
    elif case == "C":
        c = kw["c"]
        ch.event("E1", ch.add_qubits(2), [], ("fresh", lambda a: np.kron(marker(c)(a), same([0.8, 0.6])(a))))
    elif case == "D":
        phi = kw["phi"]
        ch.event("E1", ch.add_qubits(1), [], ("fresh", same([1, 0])))
        ch.event("E2", ch.add_qubits(2), ["E1"], ("fresh", lambda a: np.array([1, 0, 0, np.exp(1j * phi * a)]) / math.sqrt(2)))
        ch.event("E3", ch.add_qubits(1), ["E2"], ("fresh", same([1, 0])))
    elif case in ("E", "F"):
        cP = kw["cP"]
        ch.event("entangle", ch.add_qubits(1), [], ("fresh", marker(cP)))       # partner P
        if kw.get("measure"):
            ch.event("measure P", ch.add_qubits(1), ["entangle"], ("measure", (0, "Z" if case == "E" else "X")))
    elif case == "G":
        pass
    return ch


def end_event_past(ch, case):
    """Indices of record qubits in the end event's past: everything except a measurement of P,
    which is spacelike to the end event (cases E and F)."""
    vecs, nq = ch.branches()
    if case in ("E", "F") and nq == 2:
        return [0], nq
    return list(range(nq)), nq


def analyse(case, R=0.5, **kw):
    ch = build(case, **kw)
    vecs, nq = ch.branches()
    past, _ = end_event_past(ch, case)
    Je = fid_over(vecs, nq, past)
    Jphys = fid_over(vecs, nq, list(range(nq)))
    Jout = port_bhattacharyya(R)
    V, phase, _ = fringes(vecs, nq, R)
    Vmax = 2 * math.sqrt(R * (1 - R))
    return ch, vecs, nq, Je, Jphys, Jout, V, Vmax


def main():
    out = L.Output("toy1b", "Toy 1b: a chain of events (the reach rule)")
    out.flag("J_out of the end event", "Bhattacharyya of the detector-port distributions given each path, 2 sqrt(R(1-R))",
             "the end event's own output is the click, a distribution over ports")
    out.flag("case D placement", "E1 and E3 write path-independent qubits; E2 writes the correlated pair", "the spec gives the pair, not the chain around it")
    out.flag("case E/F partner", "P is written by an entangling event with overlap c_P (0 and 0.6); the measuring event copies P "
             "into a meter qubit (Z basis in E, X basis in F) and is spacelike to the end event", "the spec does not fix c_P")
    out.flag("record threshold", f"applied to each event's JOINT output: D >= {L.D_MIN}; R_delta from single output qubits "
             f"each holding >= (1 - {L.DELTA_R}) of the event's joint information (a lower bound), also against H_S = 1 bit; "
             "R_min in {1, 2, 5}", "adopted 17 Nov: the threshold labels events and never filters J_e")
    out.flag("inheritance domain", "an observer holding every record in the end event's past", "the spec asks for the joint quantity per domain")

    rows, chain_rows = [], []
    # A
    ch, vecs, nq, Je, Jp, Jo, V, Vm = analyse("A")
    L.check(abs(Je - 1) < TOL and abs(V - Vm) < 1e-9, "case A: J_e = 1, V = V_max")
    rows.append(["A independent", "", Je, 1.0, Jp, Jo, V, Vm])
    # B
    cs = [0.9, 0.7, 0.95, 0.8]
    ch, vecs, nq, Je, Jp, Jo, V, Vm = analyse("B", cs=cs)
    L.check(abs(Je - np.prod(cs)) < TOL, "case B: J_e = product of c_k")
    rows.append(["B dependent", f"c = {cs}", Je, float(np.prod(cs)), Jp, Jo, V, Vm])
    prev = 1.0
    for k in range(len(cs)):
        v, n = ch.branches(upto=k)
        Jk = fid_over(v, n, list(range(n)))
        L.check(Jk <= prev + TOL, "J_e never rises along the chain (B)")
        L.check(abs(Jk - np.prod(cs[: k + 1])) < TOL, "case B along the chain")
        Jout_k = float(abs(np.vdot(marker(cs[k])(0), marker(cs[k])(1))))
        chain_rows.append(["B", f"E{k+1}", Jk, float(np.prod(cs[: k + 1])), Jout_k])
        prev = Jk
    # C
    ch, vecs, nq, Je, Jp, Jo, V, Vm = analyse("C", c=0.6)
    L.check(abs(Je - 0.6) < TOL, "case C: only the dependent qubit's overlap enters")
    rows.append(["C mixed event", "c = 0.6, second qubit path-independent", Je, 0.6, Jp, Jo, V, Vm])
    # D
    for phi in np.linspace(0, math.pi, 7):
        ch, vecs, nq, Je, Jp, Jo, V, Vm = analyse("D", phi=float(phi))
        exp = abs(math.cos(phi / 2))
        sep = fid_over(vecs, nq, [1]) * fid_over(vecs, nq, [2])
        L.check(abs(Je - exp) < TOL and abs(V - Vm * exp) < 1e-9, f"case D: joint fidelity |cos(phi/2)| at phi = {phi}")
        L.check(abs(fid_over(vecs, nq, [1]) - 1) < 1e-9 and abs(fid_over(vecs, nq, [2]) - 1) < 1e-9, "case D: each qubit alone is the same")
        if phi > 0:
            L.check(sep > Je, "case D: product of separate fidelities exceeds the joint fidelity")
        rows.append(["D correlated", f"phi = {phi:.3f}; separate product = {sep:.3f}", Je, exp, Jp, Jo, V, Vm])
        prev = 1.0
        for k in range(3):
            v, n = ch.branches(upto=k)
            Jk = fid_over(v, n, list(range(n)))
            L.check(Jk <= prev + TOL, "J_e never rises along the chain (D)")
            prev = Jk
            if abs(phi - math.pi / 2) < 1e-9:
                chain_rows.append(["D (phi = pi/2)", f"E{k+1}", Jk, 1.0 if k == 0 else exp, fid_over(v, n, [n - 1]) if k != 1 else 1.0])
    # E: no signalling
    for cP in (0.0, 0.6):
        res = {}
        for meas in (False, True):
            ch, vecs, nq, Je, Jp, Jo, V, Vm = analyse("E", cP=cP, measure=meas)
            res[meas] = (Je, V)
            rows.append(["E no signalling", f"c_P = {cP}; P measured: {meas}", Je, cP, Jp, Jo, V, Vm * cP])
        L.check(abs(res[False][0] - res[True][0]) < TOL and abs(res[False][1] - res[True][1]) < 1e-9,
                "case E: J_e and V unchanged by measuring P")
    # F: eraser (P measured in X, the complementary basis)
    for cP in (0.0, 0.6):
        ch, vecs, nq, Je, Jp, Jo, V, Vm = analyse("F", cP=cP, measure=True)
        L.check(abs(V - Vm * cP) < 1e-9, "case F: overall V unchanged")
        subsets = []
        for outcome in (0, 1):
            proj = lambda v, o=outcome: (v.reshape(2, 2)[:, o]).copy()
            Vc, ph, prob = fringes(vecs, nq, 0.5, project=proj)
            subsets.append((Vc, ph))
        L.check(abs(subsets[0][1] - subsets[1][1]) % (2 * math.pi) > 1.0, "case F: the two subsets' fringes are out of phase")
        if cP == 0.0:
            L.check(abs(subsets[0][0] - 1) < 1e-9 and abs(subsets[1][0] - 1) < 1e-9, "case F: fringes recover fully within each subset")
        rows.append(["F eraser", f"c_P = {cP}; subset V = {subsets[0][0]:.3f}, {subsets[1][0]:.3f}; "
                     f"fringe peaks at phase {subsets[0][1]:.3f} and {subsets[1][1]:.3f}", Je, cP, Jp, Jo, V, Vm * cP])
    # G: unequal split
    for R in np.linspace(0.1, 0.9, 9):
        ch, vecs, nq, Je, Jp, Jo, V, Vm = analyse("G", R=float(R))
        L.check(abs(V - Vm) < 1e-9 and abs(Je - 1) < TOL, "case G: V = V_max, J_e = 1")
        rows.append(["G unequal split", f"R = {R:.1f}", Je, 1.0, Jp, Jo, V, Vm])
    # J = V/V_max recovers J_e wherever V_max > 0
    for r in rows:
        if r[7] > 0 and not r[0].startswith("E") and not r[0].startswith("F"):
            L.check(abs(r[6] / r[7] - r[2]) < 1e-8, f"J = V/V_max recovers J_e ({r[0]} {r[1]})")
    out.table("1_cases", {**HEADER, "R": "0.5 unless stated", "P(e)": "every event before the end event in the explicit DAG"},
              ["case", "setting", "J_e", "expected J_e", "J_phys", "J_out (end event)", "V", "V_max x expected J (E, F) or V_max"], rows,
              note="Visibility is not join: V = V_max J. In E and F the last column is the expected V.")
    out.table("2_along_the_chain", {**HEADER, "channel": "all records written so far (J_e at each event)"},
              ["case", "event", "J_e at the event", "expected", "J_out (that event's own outputs)"], chain_rows)

    # record threshold and inheritance (cases B and D)
    rec_rows, inh_rows = [], []
    for case, kw in (("B", {"cs": cs}), ("D", {"phi": math.pi / 2})):
        ch = build(case, **kw)
        vecs, nq = ch.branches()
        start = 0
        per_event, per_qubit = 0.0, 0.0
        for (name, outs, parents), (ei, writer) in zip(ch.events, ch.ops):
            k = len(outs)
            qs = list(range(start, start + k))
            r0, r1 = reduced(vecs[0], nq, qs), reduced(vecs[1], nq, qs)
            D = float(L.trace_distance(r0, r1))
            chi = L.holevo([0.5, 0.5], [r0, r1])
            single = lambda i: L.holevo([0.5, 0.5], [reduced(vecs[0], nq, [qs[i]]), reduced(vecs[1], nq, [qs[i]])])
            R_own = L.redundancy(single, k, chi)
            R_S = L.redundancy(single, k, 1.0)
            ev = [L.is_event(D, R_own, R_min=rm) for rm in L.R_MIN_LIST]
            rec_rows.append([case, name, k, D, chi, R_own, R_S] + ev)
            per_event += chi
            per_qubit += sum(single(i) for i in range(k))
            start += k
        joint = L.holevo([0.5, 0.5], [reduced(vecs[0], nq, list(range(nq))), reduced(vecs[1], nq, list(range(nq)))])
        inh_rows.append([case, joint, per_event, per_event - joint, per_qubit, per_qubit - joint])
        L.check(joint <= 1 + 1e-12, "inheritance never exceeds H(prior)")
    out.table("3_record_threshold", {**HEADER, "D_min": L.D_MIN, "delta": L.DELTA_R},
              ["case", "event", "output qubits", "D of the joint output", "chi bits", "R_delta (vs the event's own information)",
               "R_delta (vs H_S = 1 bit)", "event at R_min = 1", "R_min = 2", "R_min = 5"], rec_rows)
    out.table("4_inheritance", {**HEADER, "domain": "an observer holding every record in the end event's past"},
              ["case", "joint chi bits", "sum over events", "over-count (events)", "sum over record qubits", "over-count (qubits)"], inh_rows,
              note="Signed: a negative over-count means records hold more together than apart (synergy, case D).")
    for r in inh_rows:
        out.say(f"Inheritance, case {r[0]}: joint {r[1]:.4f} bits; summing per event over-counts by {r[3]:+.4f}, per record qubit by {r[5]:+.4f}")
    out.say("All checks passed: cases A-G to 1e-10; J_e never rises along a chain; J = V/V_max recovers J_e; case D's separate "
            "fidelities (1) exceed the joint |cos(phi/2)|; measuring P changes neither J_e nor V; the eraser recovers fringes "
            "within each outcome, out of phase.")
    out.finish()


if __name__ == "__main__":
    main()
