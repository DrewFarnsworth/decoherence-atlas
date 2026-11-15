# Decoherence atlas

Standard decoherence physics, reproduced against experiment, to show the shape of domains: how far, and for how long, quantum events leave parts undistinguished.

This is the first measurement project of the framework in *Quantifying the ladder* ("The core"). It adds no new physics. Its job is to map what existing physics already says, so that the framework's definitions are written against real curves.

## What it computes

For each benchmark, the record overlap the framework calls Iᵢⱼ: how far the records an event leaves would be the same whichever part took it. Physicists call this the decoherence factor. Every result goes on three common axes:

- domain size against the wavelength of the probing boson;
- domain size against temperature;
- record overlap against time.

## Rules

- **Events are boson exchanges, real or virtual alike.** The atlas models photons and their quasiparticles in matter (phonons, vibrational quanta); see `BENCHMARKS.md` for why gluons, W, Z and the Higgs are left out.
- **A benchmark passes** when the computed quantity matches the measured one within 30%, using the paper's own inputs.
- **Nothing is claimed beyond the benchmarks** until all the benchmarks pass or are explained.

## Layout

```
README.md          this file
BENCHMARKS.md      the ten targets, with sources
atlas/             shared code: constants, the scattering model, plotting
benchmarks/b1 … b10 one folder per benchmark: run.py (model, inputs, sources) and result.json
results/           scoreboard.md, atlas_points.json; the common-axis plots come next
tests/             fast checks on the shared code and each model's pieces
run_all.py         runs the benchmarks and writes the scoreboard
```

## Status

Ten benchmark scripts written (15 November 2026; B9 and B10 added that evening). See `results/scoreboard.md` for what passes, what fails and why. No new physics is claimed yet.

## Running it

You need Python 3 and four libraries. Once, in this folder:

```
pip install -r requirements.txt
```

Then:

```
python run_all.py              # all ten; B4 and B8 take a few minutes each
python run_all.py b1 b5        # just some
python run_all.py --report     # rebuild the scoreboard from saved results
python -m pytest -q            # fast checks, under a second
```

Each script prints what it computed against what was measured, and writes `result.json` beside itself. Each file's opening comment says what the event is, which model is used, which inputs come from where, and what is not yet computed.
