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

`experiments_3d_regret_plotting.ipynb` simplifies our measures into aggregates to enable 3d visualisation (section C.2 Alternative Visualisations):

- IC and CC together produce agent welfare regret (one axis)
- IA weighted by principals' magnitude gives total agent misalignment (second axis)
- Principals' welfare regret is the dependent variable of interest (third axis)
- A configurable number of games and solutions are sampled, producing a scatter plot with superimposed bound
  - CA is allowed to vary freely
  - Calibration ratios are fixed per run

`experiments_avg_regret_grid_scan.py` measures avg principals' welfare regret over a comprehensive scan of input variables (section 6.1 Empirical Validation)

- fix values for three of IC, IA, CC, and CA, while performing a sweep on the fourth
  - `run_one` performs one sweep
  - `run_experiments` scans specified fixed values and performs a sweep at each configuration
  - games are sampled randomly; calibration ratios r are free to vary within the sampling distribution
- example usagages:
  - `>>> run_experiments_avg_regret(variables=VARIABLES,sizes=[SIZES[0]],others=[0.9],increments=25,repetitions=25,name="body",progress_bar=True)`
  - `>>> run_experiments_avg_regret(variables=VARIABLES,sizes=SIZES[1:4],others=OTHERS,increments=25,repetitions=10,name="appendix",progress_bar=True)`

`experiments_inference.py` estimates the alignment and capability measures from limited observations (section 6.2 Inference of Measures)

- take limited empirical observations and infer alignment and capability measures
  - `run_one` evaluates a single configuration
  - `run_experiments` performs multiple experiments at the specified game sizes
- example usage:
  - `>>> run_experiments_inference(sizes=SIZES[:4], repetitions=25, samples=1000, increments=100, force_m=False, force_c=False, name="body")`
