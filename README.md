# B92 key-rate calculations

This directory contains numerical programs for comparing conventional and new key rates of the B92 quantum key distribution protocol used in the paper "Asymptotically tight security analysis of quantum key distribution based on universal source compression."
The programs calculate asymptotic and finite-size key rates and generate the figures used to compare them.

## Programs

| Program | Purpose | Inputs | Outputs |
| --- | --- | --- | --- |
| B92_asymptotic_conventional_new.py | Calculates conventional and new asymptotic key rates over the configured depolarizing-parameter grid. | None | B92_key_rates_asymptotic_graph.csv and B92_key_rates_asymptotic_graph.svg |
| B92_conventional_finite.py | Calculates conventional finite-size key rates. | Depolarizing parameter and output path stem | A CSV data file and PNG plot |
| B92_new_finite.py | Calculates new finite-size key rates using PICOS and QICS. | Depolarizing parameter and output path stem | A CSV data file and PNG plot |
| B92_finite_graph.py | Compares the two finite-size results with the corresponding asymptotic rates. | Depolarizing parameter | B92_finite_graph_depolarizing=DEP.svg |


## Requirements

Python 3.9 or later is recommended. The programs use:

- NumPy
- SciPy
- Matplotlib
- mpmath
- CVXPY
- Mosek
- PICOS
- QICS
- CVXOPT

Create and activate a virtual environment, then install the Python dependencies:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install numpy scipy matplotlib mpmath cvxpy mosek picos qics cvxopt

The conventional finite-size calculation and the conventional part of the asymptotic calculation use Mosek. A working Mosek installation and license are therefore required. The new finite-size calculation uses QICS through PICOS.

Versions of the Python and packages used in the paper are

- Python 3.9.13
- Numpy 1.24.3
- Scipy 1.10.1
- Matplotlib 3.7.1
- mpmath 1.3.0
- CVXPY 1.5.2
- Mosek 10.0.40
- PICOS 2.6.0
- QICS 1.1.3
- CVXOPT 1.3.2

## Usage

Run the programs from the directory where the result files should be stored.
The examples below use a depolarizing parameter of 0.01.

### 1. Calculate the asymptotic rates

    python B92_asymptotic_conventional_new.py

This produces:

- B92_key_rates_asymptotic_graph.csv
- B92_key_rates_asymptotic_graph.svg

The depolarizing-parameter grid and other analysis settings are defined near the top of the script.

### 2. Calculate the conventional finite-size rates

    python B92_conventional_finite.py 0.01 "B92_key_rate_conventional_finite_depolarizing=0.01"

This produces:

- B92_key_rate_conventional_finite_depolarizing=0.01.csv
- B92_key_rate_conventional_finite_depolarizing=0.01.png

### 3. Calculate the new finite-size rates

    python B92_new_finite.py 0.01 "B92_key_rate_new_finite_depolarizing=0.01"

This produces:

- B92_key_rate_new_finite_depolarizing=0.01.csv
- B92_key_rate_new_finite_depolarizing=0.01.png

The second argument to each finite-size program is an output path stem. Do not include .csv or .png; the extensions are appended automatically.

### 4. Create the finite-size comparison figure

    python B92_finite_graph.py 0.01

The graph program loads:

- B92_key_rate_conventional_finite_depolarizing=0.01.csv
- B92_key_rate_new_finite_depolarizing=0.01.csv
- B92_key_rates_asymptotic_graph.csv

and produces:

- B92_finite_graph_depolarizing=0.01.svg

The graph program first searches the current working directory and then the directory containing the script. It accepts finite-size CSV files with or without a header row. When the asymptotic table does not contain the requested depolarizing parameter exactly, the nearest available row is used and a warning is printed.

## Command-line inputs

Both finite-size programs require two positional arguments:

    python PROGRAM.py DEP FILE_NAME

- DEP is the depolarizing parameter and must be strictly between 0 and 1.
- FILE_NAME is the output path stem.

The graph program requires only DEP:

    python B92_finite_graph.py DEP

Use the help option to display the interface of any program that accepts command-line inputs:

    python PROGRAM.py --help

## Output data

### Asymptotic results

B92_key_rates_asymptotic_graph.csv contains five columns:

1. Depolarizing parameter
2. Conventional asymptotic key rate
3. New asymptotic key rate
4. Conventional conditional phase-error entropy
5. New conditional phase-error entropy

### Conventional finite-size results

The conventional finite-size CSV records the logarithm of the total number of rounds, key rate, bit- and phase-error entropies, worst-case empirical probabilities, and affine-region coefficients. Column names are included in the first row.

### New finite-size results

The new finite-size CSV contains:

1. Base-10 logarithm of the total number of rounds
2. Key rate
3. Bit-error entropy
4. Phase-error entropy
5. Optimized Renyi parameter

Column names are included in the first row.

## Numerical considerations

- Both finite-size programs use fixed Nelder-Mead tolerances of
  xatol = 1e-8 and fatol = 1e-8.
- The finite-size optimizations can take substantially longer than the plotting program.
- Zero key-rate points are retained in the data but are not visible on a logarithmic plot.
- The output filename must use the same decimal representation of DEP expected by B92_finite_graph.py. The commands above provide matching names.

## Troubleshooting

### A required CSV file cannot be found

Check that all three CSV files expected by B92_finite_graph.py are in the current directory or the script directory and that the depolarizing parameter in each filename matches the command-line value.

### Mosek is unavailable

Install the Mosek Python package and configure a valid license. Confirm that Mosek appears in the list returned by:

    python -c "import cvxpy as cp; print(cp.installed_solvers())"

### PICOS cannot import CVXOPT

Install CVXOPT in the same Python environment:

    python -m pip install cvxopt

### Inaccurate solver solutions

The current conventional finite-size implementation accepts only solutions reported as optimal by CVXPY/Mosek. Solutions reported as inaccurate are rejected rather than used in the key-rate calculation.

## Citation and license

When publishing results obtained with these programs, cite the associated research article and record the software version used. Before distributing the source publicly, add a repository-level LICENSE file and state the selected license here.
