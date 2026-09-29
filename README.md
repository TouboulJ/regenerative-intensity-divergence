# Robust minimum-divergence estimation of point-process intensities

Replication package for the manuscript *"Robust minimum-divergence estimation of
point-process intensities: a martingale treatment, with regenerative and self-exciting
examples"* (J. Touboul).

The method estimates a point-process intensity with a weighted-intensity
(density-power-divergence) M-estimator whose estimating equation is a stochastic
integral against the compensated counting measure. Consistency and asymptotic normality
(with the sandwich variance) are proved for a general predictable intensity via the
martingale central limit theorem; maximum likelihood is the unit-weight special case.
The paper also characterises the efficiency–robustness frontier and illustrates the
method on simulated and real event data.

## Repository layout

```
paper/
  regenerative_intensity_theory.tex     standard article class (compiles anywhere)
  regenerative_intensity_theory.pdf
  regenerative_intensity_theory_SPRINGER.tex   Springer svjour3 source (see note below)
  figures/                              figures included by the manuscript
code/
  simulations.py                        one-file driver: reproduces all figures
  intensite_phidiv.py                   renewal i.i.d. (background)
  intensite_nonrenouvellement.py        inhomogeneous Poisson            -> §(A)
  intensite_rupture.py                  change in intensity, rate n      -> §(B)
  intensite_hawkes.py                   Hawkes, sandwich variance, GOF   -> §(C)
  intensite_conjecture1.py              efficiency–robustness frontier   -> §(D)
  intensite_conditions.py               proof-condition checks (Hawkes)  -> App. F
  intensite_donnees_reelles.py          real-data illustration
```

## Requirements

Python 3.9+ with `numpy`, `scipy`, `matplotlib`, `pandas`:

```
pip install -r requirements.txt
```

## Reproduce all figures

```
python code/simulations.py
```

This writes the seven figures to `code/figures/`, one block per figure, with fixed
random seeds. To rebuild the manuscript with fresh figures, copy them next to the
`.tex`:

```
cp code/figures/*.png paper/figures/
```

Each per-result script under `code/` can also be run on its own and writes to its own
`sorties_*/` folder.

## Real data

`code/intensite_donnees_reelles.py` (and the real-data block of `simulations.py`)
download two public catalogues on first run and cache them locally:

- earthquake event times from the USGS FDSN event service,
  <https://earthquake.usgs.gov/fdsnws/event/1/>;
- the British coal-mining disaster series (`boot::coal`) from Rdatasets,
  <https://vincentarelbundock.github.io/Rdatasets/csv/boot/coal.csv>.

Without network access the code falls back to a clearly-labelled synthetic demonstration.
Efficiency is a model-based quantity and is reported only in simulation; on real data the
scripts report fit, goodness-of-fit, and sensitivity to an injected event.

## Build the manuscript

```
cd paper
pdflatex regenerative_intensity_theory.tex
pdflatex regenerative_intensity_theory.tex
```

The references are in a `thebibliography` block, so two passes resolve all
cross-references. The `regenerative_intensity_theory_SPRINGER.tex` variant uses Springer's
`svjour3` document class, which is not redistributed here; obtain it from the Springer
manuscript-template page or compile that file on Overleaf.

## Reproducibility

All randomness uses fixed seeds (`numpy.random.default_rng`), so reruns reproduce the
reported numbers up to negligible platform float differences.

## License

Code is released under the MIT License (see `LICENSE`). The manuscript text and figures
are © 2026 Jacques Touboul.

## Citation

See `CITATION.cff`. Once archived on Zenodo, cite the release DOI.
