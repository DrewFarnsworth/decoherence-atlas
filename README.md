# Decoherence atlas

Standard decoherence physics, reproduced against experiment, to show the shape of domains: how far, and for how long, quantum events leave parts undistinguished.

This is the first measurement project of the framework in *Quantifying the ladder* ("The core"). It adds no new physics. Its job is to map what existing physics already says, so that the framework's definitions are written against real curves.

## What it computes

For each benchmark, the record overlap the framework calls Iᵢⱼ: how far the records an event leaves would be the same whichever part took it. Physicists call this the decoherence factor. Every result goes on three common axes:

- domain size against the wavelength of the probing boson;
- domain size against temperature;
- record overlap against time.

## Rules

- **Events are boson exchanges.** Real or virtual (see the question in `BENCHMARKS.md`).
- **A benchmark passes** when the computed quantity matches the measured one within 30%, using the paper's own inputs.
- **Nothing is claimed beyond the benchmarks** until all eight pass or are explained.

## Layout

```
README.md          this file
BENCHMARKS.md      the eight targets, with sources
atlas/             shared code: constants, the scattering model, plotting
benchmarks/b1 … b8 one folder per benchmark: a script, its inputs, its result
results/           the common-axis plots, once there are results
```

## Status

Benchmark list drafted, awaiting review. No modelling yet.

## Running it

Python 3 with NumPy and SciPy. Spin and open-system benchmarks (B4, B7, B8) will also need QuTiP. Instructions arrive with the first script.
