# Illustrative graphics and experiments for Delegation Games

## Requirements

Python

- Python 3.11 or greater

Packages

> pip install -r requirements.txt

## Graphics

`graphics_*` notebooks generate graphics related to or used in the paper.

## Experiments

`experiments*` scripts and notebooks generate many random delegation games with various fixed/controlled measures, allowing other measures to vary. Principals' welfare regret is our main dependent variable of interest.

`experiments_3d_regret_plotting.ipynb` simplifies our measures into aggregates to enable 3d visualisation (C.2 Alternative Visualisations):

- IC and CC together produce agent welfare regret (one axis)
- IA weighted by principals' magnitude gives total agent misalignment (second axis)
- Principals' welfare regret is the dependent variable of interest (third axis)
- A confurable number of games and solutions are sampled, producing a scatter plot with superimposed bound
  - CA is allowed to vary freely
  - Calibration ratios are fixed per run

`experiments_grid_scan.py` has two modes:

1. (section 6.1 Empirical Validation)
    - fix values for three of IC, IA, CC, and CA, while performing a sweep on the fourth
       - `exp_1` performs one sweep
       - `run_exp_1` scans specified fixed values and performs a sweep at each configuration
       - games are sampled randomly; calibration ratios r are free to vary within the sampling distribution
2. (section 6.2 Inference of Measures)
     - take limited empirical observations and infer alignment and capability measures
       - `exp_2` evaluates a single configuration
       - `run_exp_2` performs multiple experiments at the specified game sizes
