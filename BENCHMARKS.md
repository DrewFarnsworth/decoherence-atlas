# Benchmarks

Draft 1, 14 November 2026. These are the published measurements the atlas must reproduce before it says anything new. Check the targets before any modelling starts.

## Rules

- **Events are boson exchanges, real or virtual alike** (decided 15 Nov 2026). Each benchmark names its boson.
- **Edges the atlas models:** photons, and the electromagnetic quasiparticles built from them in matter (phonons, vibrational quanta). At molecular energies these are the only edges that matter.
- **Edges it does not model:** gluons (they interact with each other and are confined), W and Z (they change what a particle is), the Higgs (it couples to mass). They need their own specification, and at molecular energies they are frozen: exciting them costs 10⁷ to 10¹⁰ times kT. Where one appears, as in B6, it is treated as a fixed, measured input and not as a domain edge.
- **Success is a match within 30%** to the measured quantity, using the paper's own inputs, with no fitting beyond what the paper itself fitted.
- **The quantity is always the same:** the overlap of the records an event leaves, Iᵢⱼ = |⟨Eᵢ|Eⱼ⟩|, the core's line (1). In interferometry it appears as fringe visibility, in spin experiments as the echo signal, in crystals as the recoil-free fraction, in spectroscopy as a coherence or energy-transfer time.
- **Status labels:** **[Established]** replicated and uncontested; **[Contested]** published but disputed. A † marks a number checked against the source while drafting this list.

## The organising theory

For an environment that probes a system by scattering, the record overlap between two places a distance Δx apart falls as

```
Iᵢⱼ(Δx, t) ≈ exp(−Λ Δx² t)    for Δx well below the probe's wavelength
Iᵢⱼ(Δx, t) ≈ exp(−Γ t)        beyond it, where Γ is the total scattering rate
```

with Joos and Zeh's localization rate Λ = p_g² σ_eff n_g v_g / (8π²) for gas particles of momentum p_g, effective cross-section σ_eff, density n_g and speed v_g. The small-distance form is valid only for separations well below the probe's wavelength; beyond that, the rate saturates at the collision rate † (Joos & Zeh 1985, as summarised in arXiv:1009.4349). The standard review is Schlosshauer 2019 † (Physics Reports 831:1).

**What the atlas tests:** whether every benchmark sits on this two-part shape once each is expressed as domain size against probe wavelength, temperature and time.

## The eight benchmarks

| # | System and event | Boson | What to reproduce | Tier |
|---|---|---|---|---|
| B1 | C70 fullerenes in a Talbot–Lau interferometer, decohered by collisions with background gas | Virtual photons (the van der Waals interaction) | Fringe visibility falling exponentially with gas pressure, for several gases, at pressures up to 2.5 × 10⁻⁶ mbar † | Controlled approximation |
| B2 | The same molecules heated by laser before entering, decohered by their own thermal emission | Real photons | Visibility 0.47, 0.29, 0.07 and about 0 at heating powers of 0, 3, 6 and 10.5 W †, with molecule temperatures from about 1,000 to 3,000 K † | Controlled approximation |
| B3 | Oligoporphyrins above 25,000 Da and up to 2,000 atoms, in a 2 m interferometer | None dominant: the test is that every channel stays below threshold | Fringes above 90% of the expected visibility †, superposition held for more than 7 ms † | Upper limit of the atlas |
| B4 | One electron spin (a nitrogen-vacancy centre in diamond) in a bath of carbon-13 nuclear spins at natural abundance, 1.1% † | Virtual photons (magnetic dipole and hyperfine coupling) | Spin-echo coherence time T₂ of about 100 µs to 1 ms †; the cluster-correlation method is reported to match experiment † | Exact |
| B5 | Iron-57 nuclei in α-iron absorbing 14.41 keV gamma rays without recoil (the Mössbauer effect) | Real photons and phonons | Recoil-free fraction 0.799 ± 0.003 at room temperature †, from f = exp(−k²⟨x²⟩) †, falling with temperature | Exact for a model of the lattice |
| B6 | Protons in hexagonal ice, probed by deep inelastic neutron scattering | The probe is the strong nuclear force (outside the atlas), entered as a measured scattering length; the protons' own domain is set by photons and phonons in the ice | The proton momentum distribution and mean kinetic energy against path-integral simulation †; target value to be taken from the papers | Exact for a model, with uncertain inputs |
| B7 | O–H stretch vibrations in liquid water | Vibrational quanta passed by dipole coupling (virtual photons) | Intermolecular energy transfer in 80 fs †; frequency memory lost in about 50 fs at 300 K †, lasting beyond about 200 fs at 274 K †; the excitation shared over several molecules † | Exact for a model, with uncertain inputs |
| B8 | Electronic excitations in the seven-pigment FMO photosynthetic complex | Excitons passed by dipole coupling; phonons of the protein | Electronic coherence of about 60 fs at room temperature †; at 20 K up to 200 to 500 fs, and irrelevant above about 150 K †; excitations shared over a few neighbouring pigments † | Exact for a model, with uncertain inputs. **[Contested]** history: earlier reports of picosecond coherence are now assigned to vibrations |

**In reserve:** radical pairs in robin cryptochrome 4 are magnetically sensitive in isolated protein †, but no single number yet makes a clean target. They join when one does.

## What the eight span

| Axis | Range covered |
|---|---|
| Probe wavelength | Gamma rays (B5) to thermal infrared (B2) |
| Temperature | 20 K (B8) to about 3,000 K (B2) |
| Time | Tens of femtoseconds (B7, B8) to milliseconds (B3, B4) |
| Size of what is undistinguished | A proton's spread (B6) to two path separations of a large molecule (B1 to B3) |

## Sources

- B1: Hornberger K et al. (2003). Collisional decoherence observed in matter wave interferometry. *Physical Review Letters* 90:160401. https://arxiv.org/abs/quant-ph/0303093 ; Hackermüller L et al. (2003). Decoherence in a Talbot Lau interferometer: the influence of molecular scattering. *Applied Physics B* 77:781. https://arxiv.org/abs/quant-ph/0307238
- B2: Hackermüller L et al. (2004). Decoherence of matter waves by thermal emission of radiation. *Nature* 427:711. https://arxiv.org/abs/quant-ph/0402146 ; visibilities as reported in *Physics Today*, May 2004. https://physicstoday.aip.org/news/hot-buckyballs-lose-quantum-coherence ; Hornberger K, Hackermüller L, Arndt M (2005). *Physical Review A* 71:023601.
- B3: Fein YY et al. (2019). Quantum superposition of molecules beyond 25 kDa. *Nature Physics* 15:1242. https://quantum.univie.ac.at/news/details-news/news/2000-molecules-exist-at-two-places-at-once/
- B4: Zhao N, Ho S-W, Liu R-B (2012). Decoherence and dynamical decoupling control of nitrogen-vacancy center electron spins in nuclear spin baths. *Physical Review B*. https://arxiv.org/abs/1108.2343 ; T₂ range from arXiv:1309.5921.
- B5: Lamb–Mössbauer factor of α-iron measured with a resonant spectrometer (0.799 ± 0.003). https://opact.sdu.edu.tr/vufind/SummonRecord/FETCH-LOGICAL-c312t-8dcd2cad120f33779525f7ffe13680294700d7ff3f9d08e2c6fbc51035b445473 ; formula and Debye temperature: https://hep.ph.liv.ac.uk/~jbland/JBland-Thesis-2002/node11.html
- B6: Flammini D et al. (2012). Spherical momentum distribution of the protons in hexagonal ice from modeling of inelastic neutron scattering data. *Journal of Chemical Physics*. https://math.berkeley.edu/~linlin/publications/HexExp.pdf ; Lin L et al. (2011). *Physical Review B* 83:220302(R). https://math.berkeley.edu/%7Elinlin/publications/pihex.pdf ; Senesi R et al. (2013). *Chemical Physics* 427:111.
- B7: Kraemer D et al. (2008). Temperature dependence of the two-dimensional infrared spectrum of liquid H₂O. *PNAS* 105:437. https://pmc.ncbi.nlm.nih.gov/articles/PMC2206554
- B8: Duan H-G et al. (2017). Nature does not rely on long-lived electronic quantum coherence for photosynthetic energy transfer. *PNAS* 114:8493. https://pmc.ncbi.nlm.nih.gov/articles/PMC5559008 ; low-temperature study, arXiv:2104.01462.
- Reserve: Xu J et al. (2021). Magnetic sensitivity of cryptochrome 4 from a migratory songbird. *Nature* 594:535. https://ora.ox.ac.uk/objects/uuid:c1bedca7-c96d-414a-94a2-c9a14aab18fa
- Theory: Joos E, Zeh HD (1985). *Zeitschrift für Physik B* 59:223; Schlosshauer M (2019). Quantum decoherence. *Physics Reports* 831:1. https://arxiv.org/abs/1911.06282
