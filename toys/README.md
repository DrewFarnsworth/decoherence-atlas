# Bridge toys

Seven toys that test whether one pair of formulas (distances for cut and join, information for inheritance) gives definite, consistent numbers from a which-path marker to a radical pair. The definitions are in `BRIDGE.md` and the specification in *Bridge toys: specification* (16 November 2026).

## Running them

From the repository folder:

```
python -m pytest -q toys/tests        # fast checks for every toy, about 5 seconds
python toys/run_toys.py --quick       # fast toys in full, slow toys in quick mode
python toys/run_toys.py               # everything at full settings
python toys/run_toys.py 4             # one toy
```

Each toy writes its tables (Markdown and CSV) and a one-page `SUMMARY.md` into `toys/results/toyN/`. The runner also rebuilds `toys/FLAGS.md`, the list of every modelling choice the spec did not fix.

Approximate full-run times on a laptop:

| Toy | Time |
|---|---|
| 1 markers, 2 HOM, 6 radical pair | under 30 s each |
| 5 QED scattering | about a minute |
| 3 superradiance | about 4 min |
| 4 graph domains | not yet timed; try `--quick` first |
| 7 whose parts | 30 min or more (reduced sweep; see FLAGS.md) |

## Handing results back

Send back the `SUMMARY.md` of each toy you ran, plus `toys/FLAGS.md`. Those are short and hold the key numbers. If a run stops with an error, send its last few lines.

## Rescoring the atlas

The scoreboard's scoring rules live in `atlas/scoring.py` (tiers, measurement uncertainties, hard bounds, sign). To rescore saved results without rerunning any benchmark:

```
python run_all.py --report
```

`python benchmarks/b8/scan.py` scans for one bath that gives both FMO coherence times. It runs in about a minute; add `--heom` for the slow full test.
