# HMC-Analytic
Autotuning Hamiltonian Monte Carlo for Analytical Test Functions

This repository contains a collection of Python codes and Jupyter notebooks for autotuning Hamiltonian Monte Carlo based on a factorised version of the L-BFGS method. 

Table of Contents

- Factorised BFGS.ipynb: Jupyter notebook to test the factorised L-BFGS implementation.
- Test BFGS Updates.ipynb: Jupyter notebook to test various versions of quasi-Newton updates against each other.
- Plot Test Functions.ipynb: Jupyter notebook to visualise the analytical test functions.
- Analysis 2.ipynb and Analysis.ipynb: Simplistic Jupyter notebooks to perform some basic output analyses.

- testfunctions.py: Python code that implements tests functions (potential energies, derivatives, etc.)
- samplestatistics.py: Python code to perform on-the-fly statistical calculations.

- 1-HMC-ND - Autotune BFGS.ipynb: Jupyter notebook that implements an autotuning HMC for an N-dimensional distribution based on BFGS updating of the inverse Hessian.
- 2-HMC-ND - Autotune FBFGS.ipynb: Jupyter notebook that implements an autotuning HMC for an N-dimensional distribution based on factorised BFGS (F-BFGS) updating of the (inverse) Hessian.	
- 3-HMC-ND - Autotune LFBFGS.ipynb: Jupyter notebook that implements autotuning HMC for an N-dimensional distribution based on limited-memory, factorised BFGS (LF-BFGS) updating of the inverse Hessian.
- 4-HMC-ND - Autotune LFBFGS Macro.ipynb: Jupyter notebook that implements macroscopic autotuning HMC for an N-dimensional distribution based on limited-memory, factorised BFGS (LF-BFGS) updating of the inverse Hessian.
