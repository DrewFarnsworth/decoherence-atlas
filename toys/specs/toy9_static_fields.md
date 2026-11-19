# Toy 9: records left by static fields (specification, as built)

Written 19 November 2026 to match companion revision 4.6, so that this toy survives later revisions of the
paper and of *Bridge toys: specification*. Paste this section into the specification, or edit it here; the code
is `toys/toy9_static_fields.py`.

## What it tests

Revision 4.6, primitive **Event**: whether an interaction is an event depends on the record it leaves, not on
whether its quantum is real or virtual. A static field records where it differs between the alternatives and
something registers the difference. Where the field is the same across the alternatives, as inside a bound
state, it joins.

No photon is emitted anywhere in this toy. Every record is written by a static potential acting for a time tau.

## Part A: two bodies in superposition

- Two bodies on a line, each in two arms (separations d1, d2; centres D apart), coupled only by C / r
  (Coulomb, either sign, or gravity).
- Question: which arm body 1 is in. Prior uniform.
- Phase reader (body 2 also in superposition): J = |cos(dphi/2)|, dphi = phi_00 - phi_01 - phi_10 + phi_11,
  phi_ab = -C tau / (hbar r_ab). Computed from the state vector; concurrence alongside.
- Kick reader (body 2 localised, Gaussian spread sigma_x): J = exp(-dp^2 / (8 sigma_p^2)),
  sigma_p = hbar / (2 sigma_x), impulse approximation.
- Cases: two electrons at D = 10 um, 100 um and 1 mm; an attractive pair; the Bose et al. (2017) gravity
  parameters, from memory (1e-14 kg, arms 250 um, centres 450 um, 2.5 s), and the same at 1 mm.

## Part B: a bound state read from outside

- A probe electron in two arms, R -+ a0/2 along (1, 2, 3)/sqrt14, reads for tau = 1e-11 s.
- Alternatives: identical constituents swapped (He-like); hydrogen 1s or 2s (exact densities, spherical);
  a neutral, dipole-free pair along z or x (quadrupole); an electron above or below a proton (dipole).
- R from 1.5 a0 to 1e4 a0.

## Record threshold

D = sqrt(1 - J^2) (pure records) against D_min = 0.01 in one copy (R_min = 1), as in line (0).

## Checks (all exact or analytic)

1. J = |cos(dphi/2)| and J^2 + C^2 = 1 from the state vector.
2. Attraction and repulsion leave the same join.
3. A sharper kick reader cuts more.
4. The 1s potential matches (1/r + 1/a0) e^(-2r/a0).
5. Swapping identical constituents leaves the field unchanged: dphi = 0 exactly.
6. Dipole and quadrupole differences give dphi falling as R^-3 and R^-4.
7. The 1s/2s difference vanishes exponentially outside the atom.

## First results (quick mode, 19 Nov)

- Two electrons 10 um apart, arms 1 um, 1 ns: dphi = 4.4 rad, J = 0.60; at 100 um, J = 1.0000 (no event).
- Gravity at the Bose et al. parameters: dphi = 0.31 rad, J = 0.988, above the record threshold.
- From outside, how far a bound state's alternatives are an event depends on the first multipole that
  differs: at tau = 1e-11 s, the dipole difference is an event out to 300 a0, the quadrupole difference to
  100 a0, the 1s/2s difference to 20 a0, and swapped identical constituents nowhere.

## What to revisit when the theory changes

- If the event rule changes (for example, real against virtual quanta, or what counts as registering), Part A
  is the direct test.
- If the record threshold changes (D_min, R_min, or more than one copy), only the "event" columns change.
- If "joins inside a bound state" is restated, Part B's four alternatives are the cases to keep.
- Atlas B11 (image-charge decoherence of electrons over a surface; Sonnentag & Hasselbach 2007, from memory)
  would calibrate both readers against a measured static-field cut once the paper's data are in hand.
