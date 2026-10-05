#!/usr/bin/env python3
"""Plot finite and asymptotic B92 key rates.

The only input is the depolarizing parameter. For a value DEP, the script
loads these files from the current directory:

* B92_key_rate_conventional_finite_depolarizing=DEP.csv
* B92_key_rate_new_finite_depolarizing=DEP.csv
* B92_key_rates_asymptotic_graph.csv

It writes the comparison plot to B92_finite_graph_depolarizing=DEP.svg. If the
input files are not in the current directory, the script also checks its own
directory.

Runtime dependencies: matplotlib and numpy.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np



LOGGER = logging.getLogger(__name__)

CONVENTIONAL_FILE_TEMPLATE = (
    "B92_key_rate_conventional_finite_depolarizing={}.csv"
)
NEW_FILE_TEMPLATE = "B92_key_rate_new_finite_depolarizing={}.csv"
ASYMPTOTIC_FILE_NAME = "B92_key_rates_asymptotic_graph.csv"
OUTPUT_FILE_TEMPLATE = "B92_finite_graph_depolarizing={}.svg"


def _input_file_names(dep: float) -> Tuple[str, str, str]:
    """Return the expected CSV file names for one depolarizing parameter."""

    return (
        CONVENTIONAL_FILE_TEMPLATE.format(dep),
        NEW_FILE_TEMPLATE.format(dep),
        ASYMPTOTIC_FILE_NAME,
    )


def _find_data_directory(dep: float) -> Path:
    """Find a directory containing all three required CSV files."""

    file_names = _input_file_names(dep)
    candidates = [Path.cwd(), Path(__file__).resolve().parent]
    checked = []
    for directory in candidates:
        if directory in checked:
            continue
        checked.append(directory)
        if all((directory / name).is_file() for name in file_names):
            return directory

    expected = "\n".join(
        f"  {directory / name}"
        for directory in checked
        for name in file_names
    )
    raise FileNotFoundError(
        "Could not find the required input files. Checked:\n"
        f"{expected}"
    )


def _load_numeric_csv(
    path: Path,
    minimum_columns: int,
) -> np.ndarray:
    """Load a numeric CSV, accepting files with or without a header row."""

    table = np.genfromtxt(path, delimiter=",", dtype=float)
    if table.size == 0:
        raise ValueError(f"{path} is empty")

    table = np.atleast_2d(table)
    if table.shape[1] < minimum_columns:
        raise ValueError(
            f"{path} must contain at least {minimum_columns} columns"
        )

    usable_rows = np.all(
        np.isfinite(table[:, :minimum_columns]),
        axis=1,
    )
    table = table[usable_rows]
    if table.size == 0:
        raise ValueError(f"{path} contains no usable numerical rows")
    return table


def _positive_key_rates(table: np.ndarray) -> np.ndarray:
    """Return rows that can be displayed on logarithmic axes."""

    return table[
        (table[:, 0] > 0.0)
        & (table[:, 1] > 0.0)
        & np.isfinite(table[:, 0])
        & np.isfinite(table[:, 1])
    ]


def create_plot(dep: float) -> Path:
    """Create the finite/asymptotic B92 key-rate comparison plot."""

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    })

    data_directory = _find_data_directory(dep)
    conventional_name, new_name, asymptotic_name = _input_file_names(dep)
    conventional = _load_numeric_csv(
        data_directory / conventional_name,
        minimum_columns=2,
    )
    new = _load_numeric_csv(
        data_directory / new_name,
        minimum_columns=2,
    )
    asymptotic = _load_numeric_csv(
        data_directory / asymptotic_name,
        minimum_columns=3,
    )

    closest_index = int(
        np.argmin(np.abs(asymptotic[:, 0] - dep))
    )
    selected_dep = float(asymptotic[closest_index, 0])
    if not np.isclose(selected_dep, dep):
        LOGGER.warning(
            "No exact asymptotic row for dep=%s; using dep=%s",
            dep,
            selected_dep,
        )

    try:
        plt.style.use("seaborn-v0_8-colorblind")
    except OSError:
        LOGGER.debug("The seaborn colorblind style is unavailable")

    figure, axis = plt.subplots()
    axis.set_title(
        f"Comparison of key rates at {dep:.1%} depolarizing"
    )

    round_grid = 10.0 ** np.linspace(5.0, 10.0, 10)
    axis.set_xticks(
        round_grid,
        [f"{round_count:e}" for round_count in round_grid],
    )
    # Alternative range retained for dep = 0.045:
    #axis.set_xlim(10.0**6, 10.0**10)
    axis.set_xlim(10.0**5, 10.0**10)
    axis.set_xlabel("Number of communication rounds")
    axis.set_xscale("log")

    key_rate_ticks = [10.0 ** -(index + 1) for index in range(5)]
    axis.set_yticks(
        key_rate_ticks,
        [f"{key_rate:.1e}" for key_rate in key_rate_ticks],
    )
    axis.set_ylim(10.0**-4, 10.0**-1)
    axis.set_ylabel("Key rate")
    axis.set_yscale("log")

    new_positive = _positive_key_rates(new)
    conventional_positive = _positive_key_rates(conventional)
    if new_positive.size:
        axis.scatter(
            10.0 ** new_positive[:, 0],
            new_positive[:, 1],
            marker=".",
            s=25,
            label="New (finite)",
        )
    if conventional_positive.size:
        axis.scatter(
            10.0 ** conventional_positive[:, 0],
            conventional_positive[:, 1],
            marker=".",
            s=25,
            label="Conventional (finite)",
        )

    axis.plot(
        round_grid,
        np.full(round_grid.shape, asymptotic[closest_index, 2]),
        "--",
        label="New (asympt.)",
    )
    axis.plot(
        round_grid,
        np.full(round_grid.shape, asymptotic[closest_index, 1]),
        "--",
        label="Conventional (asympt.)",
    )
    axis.legend(loc="lower right")
    figure.tight_layout()

    output_path = (
        data_directory / OUTPUT_FILE_TEMPLATE.format(dep)
    )
    figure.savefig(output_path)
    plt.close(figure)
    LOGGER.info("Wrote %s", output_path)
    return output_path


def parse_arguments(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """Parse the sole public input parameter."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dep",
        type=float,
        help="Depolarizing parameter (strictly between 0 and 1).",
    )
    return parser.parse_args(argv)


def main(dep: float) -> int:
    """Load the expected CSV files and write their comparison plot."""

    if not 0.0 < dep < 1.0:
        raise ValueError("dep must be strictly between 0 and 1.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        create_plot(dep)
    except (OSError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    arguments = parse_arguments()
    raise SystemExit(main(arguments.dep))
