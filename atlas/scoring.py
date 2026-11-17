"""Scoring rules for the scoreboard (Bridge toys specification, 'Atlas scoring revisions').

Applied by run_all.py to saved result.json files, so rows can be rescored without rerunning.

- Tier 1: the atlas against the published theory. Tier 2: published theory against experiment.
  Tier 3: a prediction specific to the framework (none yet). Scores are reported per tier.
- A stated measurement uncertainty is used where one exists: pass if |computed - measured| <= K_SIGMA * err.
  The 30% rule applies only where no uncertainty is stated, and those rows are marked.
- Bounds are hard: a row passes only if the computed value satisfies the bound (within its uncertainty
  if one is stated). Outside the bound but within 30% is a 'flag', not a pass.
- Sign: whether the computed decoherence is too strong or too weak, where that is meaningful.
- Rows true by construction, sensitivity runs and negative controls are info, not scored.
"""

K_SIGMA = 2.0          # flagged choice: two standard uncertainties
TOL = 0.30

# (benchmark, start of the row name) -> metadata. First match wins.
# more: which direction of the computed value means MORE decoherence ("higher" or "lower"); None = not a decoherence quantity
META = [
    ("B1", "p0 against the paper's theory", dict(tier=1, more="lower")),
    ("B1", "decoherence pressure p0", dict(tier=2, more="lower", err_from_note=True)),
    ("B2", "visible photons", dict(tier=2, more="higher")),
    ("B2", "V/V0", dict(tier=2, more="lower")),
    ("B2", "temperature at the first grating", dict(tier=2, more=None)),
    ("B2", "visibility at", dict(tier=2, more="lower")),
    ("B3", "record overlap left", dict(tier=2, more="lower")),
    ("B3", "thermal-photon scattering exponent", dict(info="side result")),
    ("B4", "Hahn-echo T2", dict(tier=2, more="lower")),
    ("B5", "recoil-free fraction", dict(tier=2, more="lower", err=0.003)),
    ("B5", "same, with theta", dict(info="sensitivity")),
    ("B6", "proton mean kinetic energy", dict(tier=2, more=None, err=2.0)),
    ("B6", "against the path-integral", dict(tier=1, more=None, err=2.0)),
    ("B7", "anisotropy decay", dict(tier=2, more="lower")),
    ("B7", "Foerster", dict(info="limit")),
    ("B7", "coherent short-time", dict(info="limit")),
    ("B7", "loss of frequency memory", dict(tier=2, more="lower")),
    ("B8", "optical dephasing time at room", dict(tier=2, more="lower")),
    ("B8", "inter-exciton coherence time at room", dict(tier=2, more="lower")),
    ("B8", "inter-exciton coherence time at 20 K", dict(tier=2, more="lower")),
    ("B8", "", dict(info="sensitivity")),
    ("B9", "HOM dip width", dict(tier=2, more=None)),
    ("B9", "HOM dip visibility", dict(tier=2, more="lower", err=0.0203)),
    ("B9", "two-photon visibility, quantum dot", dict(info="negative control: T2/2T1 includes slow drifts (see toy 2, case c)")),
    ("B10", "fringe visibility at V_EOM = 150", dict(tier=2, more="lower", err=0.02)),
    ("B10", "fringe visibility at V_EOM = 40", dict(tier=2, more="lower", err=0.02)),
    ("B10", "fringe visibility at V_EOM = 0", dict(tier=2, more="lower")),
    ("B10", "V^2 + D^2", dict(info="true by construction for pure states")),
]

NOTES = {
    "B1": "Atlas-to-paper ratio 1.21 +/- 0.03 for all nine gases: a global prefactor or convention, not yet found (needs the full paper).",
    "B2": "Visibility against heating power still needs the temperature distribution (Hornberger 2005). The emission model already "
          "resolves the which-path record by photon wavelength against path separation (1 - sinc).",
    "B4": "Overshoot grows as the bath thins (x1.4 at 1.1%, x1.8 at 0.3%): not yet investigated.",
    "B7": "The scored 79 fs is the five-molecule exciton cluster with static disorder. Frequency-memory loss needs molecular dynamics: not computed.",
    "B8": "One bath for both times: run benchmarks/b8/scan.py locally (optical dephasing scan is quick; the HEOM scan takes long).",
    "B9": "The quantum-dot rows are negative controls; a toy-2 case-c prediction needs independent spectral-diffusion data.",
}


def meta_for(bench, name):
    for b, prefix, m in META:
        if b == bench and name.startswith(prefix):
            return m
    return {"tier": 2, "more": None}


def score(bench, c):
    """Return (status, rule, sign, tier) for one saved check dict."""
    m = meta_for(bench, c["name"])
    x = c.get("computed")
    if "info" in m:
        return f"info ({m['info']})", "", "", "info"
    if x is None:
        return "not computed", "", "", m.get("tier", 2)
    err = m.get("err")
    if m.get("err_from_note") and "measured ±" in (c.get("note") or ""):
        try:
            err = float(c["note"].split("measured ±")[1].split()[0].strip(";,"))
        except (ValueError, IndexError):
            err = None
    meas, lo, hi = c.get("measured"), c.get("measured_low"), c.get("measured_high")
    sign = ""
    if meas is not None:
        if err is not None:
            ok, rule = abs(x - meas) <= K_SIGMA * err, f"{K_SIGMA:g} sigma"
        else:
            ok, rule = abs(x - meas) <= TOL * abs(meas), "30% (no uncertainty stated)"
        status = "pass" if ok else "fail"
        if not ok and m.get("more"):
            sign = _sign(x > meas, m["more"])
    else:
        e = err or 0.0
        below = lo is not None and x < lo - e
        above = hi is not None and x > hi + e
        rule = "hard bound" + (f" (+/- {e:g})" if e else "")
        if not (below or above):
            status = "pass"
        else:
            near = (below and x >= lo * (1 - TOL)) or (above and x <= hi * (1 + TOL))
            status = "flag" if near else "fail"
            if m.get("more"):
                sign = _sign(above, m["more"])
    return status, rule, sign, m.get("tier", 2)


def _sign(computed_high, more):
    stronger = computed_high if more == "higher" else not computed_high
    return "too strong" if stronger else "too weak"
