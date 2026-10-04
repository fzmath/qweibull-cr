# qweibull-cr

Reproducible research package for:

> **q-Weibull Competing Risks under Weak Identifiability: Bayesian Inference via the No-U-Turn Sampler**

Authors: Huirong Cao (Langfang Normal University, Langfang) and Fuchang Wang (University of Emergency Management, Sanhe). Corresponding author: Huirong Cao (huirongcao@126.com). Target journal: *Quality and Reliability Engineering International* (QREI, Wiley).

## Repository layout

```
qweibull-cr/
  code/     Python (numpy/scipy/pandas/matplotlib/pymc) scripts
  data/    voltage.csv, simulation result JSONs, posterior quantities
  figures/ fig1..fig10 (PDF vector + PNG)
  paper/   main.tex, references.bib, compiled main.pdf, USG.cls, build support
```

## Main scripts

- `qweibull.py`, `bayes_model.py`, `cr_bayes_model.py` — q-Weibull likelihood, cause-specific competing-risk log-posterior and analytic gradient.
- `nuts.py` — NUTS / HMC sampler with Hessian-based preconditioning and smooth-barrier wall.
- `penalized_mle.py` — unpenalised MLE vs penalised MLE / MAP decomposition.
- `fisher_info.py` — Fisher information matrix, eigenvalues and condition number across truncation designs.
- `param_compare.py` — quantitative comparison of the (q, log η, log β) vs (q, log λ, log β) parameterisations.
- `pymc_baseline.py` — mature-library (PyMC/ArviZ) reference fit.
- `sim_eta_beta.py`, `prior_center_sensitivity.py`, `boundary_fraction.py`, `sample_size.py` — recovery and sensitivity studies.
- `ppc_voltage.py`, `dependent_cr.py` — posterior predictive check and dependent-competing-risks misspecification experiment.
- `run_voltage_bayes.py`, `run_voltage_bayes_weibull.py`, `verify_cif.py`, `compute_posterior_quantities.py` — real-data analysis.

## Reproduce

1. Install Python 3.10+ with `numpy scipy pandas matplotlib` (and `pymc arviz` for the external baseline).
2. Run the scripts in `code/` to regenerate the JSON results in `data/` and the figures in `figures/`.
3. Compile `paper/main.tex` with **XeLaTeX** (the Wiley NJD/USG class loads local STIX fonts in `paper/Fonts/`):
   `xelatex → bibtex → xelatex → xelatex`, or run `paper/build_pdf.ps1`.

The `voltage` data are derived from the `weibulltools` R package and are included as `data/voltage.csv`.

## License

Code is released for reproducibility under the MIT License; the paper text and figures are © the authors.
