# Reproduction code

This package reproduces the numerical experiments in *Contamination-Resistant Adaptive Identification for Robot Manipulators Under Recoverable Actuator Effectiveness Loss*. It includes the proposed latched dual-calibrated nominal correction (LDCNC), the internal baselines, five matched-interface comparison methods, and the seven-degree-of-freedom MuJoCo manipulator used in the manuscript.

## Environment

Create and activate the supplied Conda environment:

```bash
conda env create -f environment.yml
conda activate control
```

The estimator uses CUDA automatically when a compatible device is available; all experiments can also run on CPU.

## Numerical experiments

Run an experiment group with one of the following commands:

```bash
python scripts/run_reproduction.py internal
python scripts/run_reproduction.py pulse
python scripts/run_reproduction.py profiles
python scripts/run_reproduction.py severity
python scripts/run_reproduction.py robustness
python scripts/run_reproduction.py sensitivity
```

The `internal` group evaluates the commanded-torque, snapshot, history-point, fixed-deadband, structural-ablation, proposed, and oracle interfaces. The `all` option runs every group. The scripts write their JSON records to `outputs/`. Seeds, scenario counts, fault profiles, and perturbation settings are fixed by the experiment entry points. Divergent runs remain in the records and paired methods use the same scenario realization.

The supplied `reference_results/` records are the data used to construct the manuscript tables and figures. Rebuild those assets with the following command. Newly generated records in `outputs/` take precedence over the supplied records with the same name.

```bash
python scripts/run_reproduction.py figures
```

The generated LaTeX tables and PDF/PNG figures are written to `outputs/paper_assets/`.

## License

The code and model files are released under the MIT License. The cited publications remain subject to their respective publishers' terms.
