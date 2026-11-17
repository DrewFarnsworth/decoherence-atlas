# The bridge: one pair of formulas from quantum records to neurons

Draft, 16 November 2026. The code is `toys/bridge.py` (runs in a few seconds; it checks its own arithmetic).

## Definitions

**What has to be stated before anything is computed**

- **The event e and its alternatives a.** What could have happened differently: which path, which synapse, which molecule.
- **The prior p(a).** How likely each alternative is taken to be. Both toys use a uniform prior. That is a choice, like IIT's uniform intervention, and other priors give other numbers.
- **The records ρₐ.** What each alternative leaves in the rest of the world: a quantum state, or a probability distribution if the record is classical.
- **The domain X and its channel Φ_X.** How X registers records, coarse-grained to its threshold (line 0). X sees Φ_X(ρₐ), not ρₐ.

**Cut and join: distances.** For two alternatives i and j, at X's grain:

```
J_ij^X = F(Φ_X ρ_i, Φ_X ρ_j)                  join: root fidelity (Bhattacharyya coefficient if classical)
D_ij^X = ½ ‖Φ_X ρ_i − Φ_X ρ_j‖₁                cut: trace distance (total variation if classical)
1 − J ≤ D ≤ √(1 − J²)   so   J² + D² ≤ 1      Fuchs–van de Graaf
```

For pure records, J = |⟨Eᵢ|Eⱼ⟩| and J² + D² = 1, the core's line (1). Fidelity never falls under coarse-graining, so a coarser X finds more joined.

**Inheritance: information.** What e delivers to R:

```
E(e → R) = χ({p_a, Φ_R ρ_a})      Holevo information; mutual information I(A; s_R) if classical
```

Information can only be lost along a chain of channels, so E can never exceed what the event made, however many paths there are. This replaces "multiplies along each path and adds across paths."

**Converting between them.** For two equally likely pure records with overlap c, so that D = √(1 − c²):
- the most information the records hold is χ = h((1 + c)/2) bits, where h is the binary entropy;
- the most any measurement can get out is 1 − h((1 + D)/2) bits.

Synergy ("sharing what none holds alone") is not used in these formulas. Measures of synergy disagree with each other, so it stays a qualitative description until one is chosen.

## Toy 1: double slit with an adjustable which-path record

The photon's two paths leave a marker in states with overlap c. X reads the marker through noise of strength q.

| c | q | J | D | χ (bits) | Accessible (bits) |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 1 | 1 | 1 |
| 0 | 0.9 | 0.995 | 0.10 | 0.007 | 0.007 |
| 0.42 | 0 | 0.42 | 0.91 | 0.87 | 0.73 |
| 0.42 | 0.9 | 0.996 | 0.09 | 0.006 | 0.006 |
| 0.9 | 0 | 0.90 | 0.44 | 0.29 | 0.14 |

- **The B10 settings fit.** Treat the output port as the marker, with c equal to the visibility 2√(R(1−R)). Then R = 0.43, 0.045 and 0 give J = 0.99, 0.42 and 0; D = 0.14, 0.91 and 1; and χ = 0.04, 0.87 and 1 bit.
- **Join depends on grain.** The fringes on the screen stay at c, because they answer to the record as it exists anywhere. An X that reads the marker through noise finds the two paths joined: at q = 0.9, J = 0.996 even when c = 0. Both statements hold at once, at different grains. This is the core's line (1) made relative to X.
- **The bound is strict when it should be.** Reading the two records through different amounts of noise gives J² + D² = 0.77, below 1, as Fuchs–van de Graaf allows.

## Toy 2: a neuron summing N inputs

The event is one input spike. The alternatives are which of N synapses it came through, with a uniform prior. The toy values, which are stated rather than measured:
- EPSP sizes drawn from a lognormal distribution, median 0.5 mV;
- noise at the soma of 1 mV;
- an output spike when the soma passes 1.5 mV.

| N | At the synapse's grain | At the soma's grain: mean J | Mean D | I(which; V) bits | At the spike's grain: I(which; spike) bits | log₂ N |
|---|---|---|---|---|---|---|
| 2 | J = 0, D = 1 | 0.41 | 0.82 | 0.68 | 0.56 | 1 |
| 10 | J = 0, D = 1 | 0.79 | 0.27 | 0.49 | 0.26 | 3.3 |
| 100 | J = 0, D = 1 | 0.93 | 0.21 | 0.21 | 0.11 | 6.6 |
| 1000 | J = 0, D = 1 | 0.92 | 0.23 | 0.26 | 0.14 | 10.0 |

- **Cut at their own grain, joined at the soma's.** Each synapse records its own release (J = 0). At the soma, most pairs of synapses are mostly joined: mean J is about 0.8 to 0.9 from N = 10 up.
- **The soma keeps very little about which synapse.** It holds a few tenths of a bit, out of the 3 to 10 bits that exist at the synapses. The output spike keeps less again, as data processing requires.
- **No over-counting.** Suppose the same input reaches the soma by two branches with independent noise. The two together deliver 0.51 bits, against 0.49 for one. Adding across paths would claim 0.98 bits.
- **Toy-level wobble.** The N = 100 and N = 1000 rows differ only because the weights are drawn at random. They are illustrations, not data.

## What this shows

One pair of formulas gives definite numbers in both toys: fidelity and trace distance for cut and join, Holevo or mutual information for inheritance. The quantum case and the classical case differ only in what the records are. So the bridge holds at the level of definitions. Whether it holds for real neurons is the §7.3 test, run with measured EPSP distributions and noise.

## Still open

- **The prior and the choice of parts** now decide the numbers, and so decide which domain comes out as the subject. They have to be stated, and ideally fixed by a principle.
- **Frame dependence (point 2).** Valence needs the tension T on both sides of a single event. For a Compton event, ΔT is either frame-dependent or zero.
- **Events with no temperature (point 3).** The barrier form of line (0) still contains T. A photon in a vacuum double slit needs another threshold, perhaps h/4 of action per step together with the record's lifetime.
- **How far point 4 reaches.** If an event is defined by the record it leaves, several sentences change:
  - §1.1 and §2.1 ("one exchange of a boson, real or virtual");
  - §2.4 ("structures are domains that events keep re-forming");
  - §4.1 ("a proton is mostly such events").

  Barrier-held domains, such as atoms and protons, then persist without events, and only upkeep-held domains need re-forming. Förster transfer stays an event, because the acceptor's excitation is a real record.
- **Novelty.** Quantum versions of integrated information already exist: Zanardi, Tomka and Venuti 2018, and the formalisation by Kleiner and Tull. Both are cited from memory and unchecked. What stays distinct here is the event ontology, the threshold, and valence tied to tension.
