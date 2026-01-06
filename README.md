# Standard-QP Reformulation for SVM+ (LUPI) — MATLAB Code

This repository contains the MATLAB implementation and experimental scripts supporting the paper on an **exact reformulation of the SVM+ (LUPI) optimization problem into a standard Quadratic Programming (QP) form**. The goal is to enable efficient training using mature QP solvers (e.g., `quadprog`) while preserving **full equivalence** with the classical SVM+ formulation solved via symbolic/analytical approaches (referred to here as `solve`).

The repository also includes:
- Preparation pipelines to build **privileged information (PI)** for two real-world benchmarks: **MNIST** and **CWRU Bearing**.
- **GAGS**, a parallelized genetic-algorithm-based grid search to optimize SVM+ hyperparameters.
- Experiment scripts used to generate the results reported in **Chapter 5** of the manuscript.

---

## Contents

### MNIST: Privileged Information Preparation (a1–a5)
Scripts `a1_*` to `a5_*` implement the MNIST pipeline used to generate privileged information and reduced representations.

- `a1_Dataset_converter.m`  
  Converts/loads MNIST into the internal format used by the project.

- `a2_Dataset_reduced.m`  
  Builds a reduced MNIST dataset (e.g., subset selection and/or dimensionality reduction).

- `a3_showImages.m`  
  Utility script to visualize MNIST samples.

- `a4_reducedKmeans.m`  
  Performs clustering (K-means) on the reduced MNIST representation.

- `a5_featuresPI.m`  
  Computes/exports the **privileged-information features** for MNIST (PI pipeline output).

---

### CWRU Bearing: Privileged Information Preparation
Scripts prefixed with `CWRU_` generate and visualize privileged information for the Case Western Reserve University bearing dataset.

- `CWRU_generaX.m`  
  Generates feature matrices for CWRU, including the standard feature space and the privileged-information feature space.

- `CWRU_plot_signals.m`  
  Plotting/inspection utilities for vibration signals and derived features.

---

### SVM+ Model Optimization (GAGS)
- `GAGS.m`  
  Implements **GAGS** (Genetic Algorithm-based Grid Search), used to select the best SVM+ hyperparameters under the LUPI paradigm.  
  The GA evaluates candidate configurations through the fitness functions described below.

---

### Fitness Functions (Classical vs QP Reformulation)
Two alternative fitness functions are provided depending on how the SVM+ optimization is solved:

- `FOM_LUPI.m`  
  Fitness function using the **classical SVM+ solution** computed via the symbolic/analytical baseline (`solve`).

- `QFOM_LUPI.m`  
  Fitness function using the **standard QP reformulation**, solved via a QP solver (e.g., `quadprog`).

---

### Experiments and Reproducibility (Chapter 5)
These scripts reproduce the experiments and consistency checks presented in Chapter 5 of the manuscript.

- `Pruebas_cuadraticas_basica.m`  
  Core consistency tests for the quadratic reformulation (e.g., random instances, objective checks, numerical equivalence).

- `pruebas_Quadprog.m`  
  Benchmarking and performance tests using `quadprog` (timings, scaling with sample size, etc.).

---

## Requirements

- MATLAB R2020a or newer (recommended)
- Optimization Toolbox (required for `quadprog` and `fmincon`)
- Symbolic Math Toolbox (only required if you run the `solve`-based baseline comparisons)

---

## Quick Start (Recommended Run Order)

### A) Reproduce synthetic consistency and efficiency experiments (Chapter 5 – solver comparison)
1. Run the basic quadratic consistency checks:
   ```matlab
   Pruebas_cuadraticas_basica
