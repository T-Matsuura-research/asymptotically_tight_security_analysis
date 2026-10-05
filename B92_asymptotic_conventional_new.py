"""Compare conventional and new asymptotic key rates for the B92 protocol.

The script evaluates both parameter-estimation methods over a range of
depolarizing parameters. It writes the numerical results to CSV and plots the
key rates as an SVG figure.

The conventional optimization uses CVXPY with the MOSEK solver. The new
procedure uses SciPy's Nelder-Mead optimizer.
"""

from pathlib import Path

import cvxpy as cp
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.stats import entropy


# Analysis settings
DEPOLARIZING_PARAMETERS = np.linspace(0.09, 0.0, 150, endpoint=False)
ERROR_CORRECTION_EFFICIENCY = 1.0
ALPHA = 0.38
BETA = np.sqrt(1 - ALPHA**2)
GAIN = 1.0
OUTPUT_STEM = "B92_key_rates_asymptotic_graph"


# Measurement operators
# Filter: measure with |phi_bar_j><phi_bar_j| * GAIN / 2.
FILTER_OPERATOR = GAIN * np.array(
    [
        [ALPHA**2, 0, 0, 0],
        [0, BETA**2, 0, 0],
        [0, 0, ALPHA**2, 0],
        [0, 0, 0, BETA**2],
    ]
)

BIT_ERROR_OPERATOR = GAIN * np.array(
    [
        [ALPHA**2 / 2, -ALPHA * BETA / 2, 0, 0],
        [-ALPHA * BETA / 2, BETA**2 / 2, 0, 0],
        [0, 0, ALPHA**2 / 2, -ALPHA * BETA / 2],
        [0, 0, -ALPHA * BETA / 2, BETA**2 / 2],
    ]
)

Z_OPERATOR = np.array(
    [
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ]
)


def build_conventional_problem():
    """Build the reusable CVXPY problem for the conventional analysis."""
    elements = cp.Variable(2, name="density_operator_elements")
    measured_rates = cp.Parameter(4, name="measured_rates")

    constraints = [
        elements[0] >= 0,
        elements[0] + measured_rates[0] >= 0,
        measured_rates[1] - elements[0] >= 0,
        measured_rates[2] - elements[0] >= 0,
        cp.geo_mean(cp.hstack([elements[0], measured_rates[0] + elements[0]]))
        >= cp.abs(elements[1]),
        cp.geo_mean(
            cp.hstack(
                [
                    measured_rates[1] - elements[0],
                    measured_rates[2] - elements[0],
                ]
            )
        )
        >= cp.abs(measured_rates[3] - elements[1]),
    ]

    density_operator = cp.bmat(
        [
            [elements[0], elements[1], 0, 0],
            [elements[1], measured_rates[0] + elements[0], 0, 0],
            [
                0,
                0,
                measured_rates[1] - elements[0],
                measured_rates[3] - elements[1],
            ],
            [
                0,
                0,
                measured_rates[3] - elements[1],
                measured_rates[2] - elements[0],
            ],
        ]
    )

    # Joint phase/bit-error probabilities, using the project's convention:
    # np_nb: no phase error, no bit error
    # yp_nb: phase error, no bit error
    # np_yb: no phase error, bit error
    # yp_yb: phase error, bit error
    np_nb_probability = cp.trace(
        density_operator[0:2, 0:2]
        @ (FILTER_OPERATOR[0:2, 0:2] - BIT_ERROR_OPERATOR[0:2, 0:2])
    )
    yp_nb_probability = cp.trace(
        density_operator[2:4, 2:4]
        @ (FILTER_OPERATOR[2:4, 2:4] - BIT_ERROR_OPERATOR[2:4, 2:4])
    )
    np_yb_probability = cp.trace(
        density_operator[0:2, 0:2] @ BIT_ERROR_OPERATOR[0:2, 0:2]
    )
    yp_yb_probability = cp.trace(
        density_operator[2:4, 2:4] @ BIT_ERROR_OPERATOR[2:4, 2:4]
    )

    event_probabilities = cp.hstack(
        [
            np_nb_probability,
            yp_nb_probability,
            np_yb_probability,
            yp_yb_probability,
        ]
    )
    probability_totals = cp.hstack(
        [
            np_nb_probability + yp_nb_probability,
            np_nb_probability + yp_nb_probability,
            np_yb_probability + yp_yb_probability,
            np_yb_probability + yp_yb_probability,
        ]
    )
    objective = cp.Maximize(
        -cp.sum(cp.rel_entr(event_probabilities, probability_totals))
    )

    problem = cp.Problem(objective, constraints)
    return measured_rates, density_operator, problem


def build_density_operator(elements, filtered_rate, minus_rate, bit_error_rate):
    """Construct the density operator represented by two free elements."""
    element_3 = 1 - minus_rate - elements[0]
    element_1 = (
        minus_rate
        + elements[0]
        - (BETA**2 - filtered_rate) / (BETA**2 - ALPHA**2)
    )
    element_2 = (
        (BETA**2 - filtered_rate) / (BETA**2 - ALPHA**2) - elements[0]
    )
    element_2_3 = (
        (filtered_rate - 2 * bit_error_rate) / (2 * ALPHA * BETA) - elements[1]
    )

    diagonal = np.diag(np.array([elements[0], element_1, element_2, element_3]))
    off_diagonal = np.array(
        [
            [0, elements[1], 0, 0],
            [elements[1], 0, 0, 0],
            [0, 0, 0, element_2_3],
            [0, 0, element_2_3, 0],
        ]
    )
    return diagonal + off_diagonal


def key_rate_conventional(elements, args):
    """Evaluate the conventional key-rate objective for SciPy optimizers."""
    filtered_rate, minus_rate, bit_error_rate = args

    if elements[0] <= 0 or elements[0] > 1:
        return 10

    density_operator = build_density_operator(
        elements, filtered_rate, minus_rate, bit_error_rate
    )
    if (np.linalg.eigvalsh(density_operator) < np.zeros(4)).any():
        return 10

    np_yb_probability = np.trace(
        density_operator[0:2, 0:2] @ BIT_ERROR_OPERATOR[0:2, 0:2]
    )
    yp_yb_probability = np.trace(
        density_operator[2:4, 2:4] @ BIT_ERROR_OPERATOR[2:4, 2:4]
    )
    np_nb_probability = (
        np.trace(density_operator[0:2, 0:2] @ FILTER_OPERATOR[0:2, 0:2])
        - np_yb_probability
    )
    yp_nb_probability = (
        np.trace(density_operator[2:4, 2:4] @ FILTER_OPERATOR[2:4, 2:4])
        - yp_yb_probability
    )

    # This expression intentionally preserves the original numerical formula.
    return (
        np_yb_probability
        * np.log2(np_yb_probability / (np_yb_probability + yp_yb_probability))
        + yp_yb_probability
        * np.log2(np_yb_probability / (np_yb_probability + yp_yb_probability))
        + np_nb_probability
        * np.log2(
            np_nb_probability
            / (np_nb_probability + yp_nb_probability)
            + yp_nb_probability
            * np.log2(
                yp_nb_probability / (np_nb_probability + yp_nb_probability)
            )
        )
    ) / filtered_rate


def key_rate_new(elements, args):
    """Evaluate the new key-rate objective for SciPy optimizers."""
    filtered_rate, minus_rate, bit_error_rate = args

    if elements[0] <= 0 or elements[0] > 1:
        return 10

    density_operator = build_density_operator(
        elements, filtered_rate, minus_rate, bit_error_rate
    )
    if (np.linalg.eigvalsh(density_operator) < np.zeros(4)).any():
        return 10

    filter_matrix = GAIN * np.diag([ALPHA, BETA, ALPHA, BETA])
    unnormalized_state = filter_matrix @ density_operator @ filter_matrix
    normalized_state = unnormalized_state / np.trace(unnormalized_state)

    eigenvalues, _eigenvectors = np.linalg.eigh(normalized_state)
    averaged_state = (
        normalized_state + Z_OPERATOR @ normalized_state @ Z_OPERATOR
    ) / 2
    averaged_eigenvalues, averaged_eigenvectors = np.linalg.eigh(averaged_state)

    return (
        -entropy(eigenvalues, base=2)
        - 1
        - np.trace(
            averaged_eigenvectors.conj().T
            @ normalized_state
            @ averaged_eigenvectors
            @ np.diag(np.log2(averaged_eigenvalues))
        )
    )


def calculate_key_rates():
    """Calculate the conventional and new key rates for every parameter."""
    measured_rates, conventional_density_operator, conventional_problem = (
        build_conventional_problem()
    )
    results = []

    for depolarizing_parameter in DEPOLARIZING_PARAMETERS:
        filtered_rate = GAIN * (
            2 * ALPHA**2 * BETA**2 * (1 - depolarizing_parameter)
            + depolarizing_parameter / 2
        )
        bit_error_rate = GAIN * depolarizing_parameter / 4
        minus_rate = ALPHA**2

        expected_density_operator = GAIN * (
            1 - depolarizing_parameter
        ) * np.array(
            [
                [BETA**2, ALPHA * BETA, 0, 0],
                [ALPHA * BETA, ALPHA**2, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ) + depolarizing_parameter * np.diag(
            [BETA**2, ALPHA**2, ALPHA**2, BETA**2]
        ) / 2
        initial_guess = np.array(
            [expected_density_operator[0, 0], expected_density_operator[0, 1]]
        )

        measured_rates.value = np.array(
            [
                minus_rate
                - (BETA**2 - filtered_rate) / (BETA**2 - ALPHA**2),
                (BETA**2 - filtered_rate) / (BETA**2 - ALPHA**2),
                1 - minus_rate,
                (filtered_rate - 2 * bit_error_rate) / (2 * ALPHA * BETA),
            ]
        )
        conventional_problem.solve(verbose=False, solver=cp.MOSEK)

        new_result = minimize(
            key_rate_new,
            x0=initial_guess,
            args=[filtered_rate, minus_rate, bit_error_rate],
            method="Nelder-Mead",
            options={"maxiter": 1000, "fatol": 1e-6},
        )

        if not new_result.success:
            print("Optimization fails!")
            results.append([depolarizing_parameter, 0.0, 0.0, 0.0, 0.0])
            continue

        new_key_rate = max(
            [
                filtered_rate
                * (
                    1
                    + new_result.fun
                    - ERROR_CORRECTION_EFFICIENCY
                    * entropy(
                        [
                            bit_error_rate / filtered_rate,
                            1 - bit_error_rate / filtered_rate,
                        ],
                        base=2,
                    )
                )
                / 3,
                0.0,
            ]
        )
        new_density_operator = build_density_operator(
            new_result.x, filtered_rate, minus_rate, bit_error_rate
        )

        if conventional_problem.status == "optimal":
            conventional_key_rate = max(
                [
                    filtered_rate
                    * (
                        1
                        - conventional_problem.value
                        / (np.log(2) * filtered_rate)
                        - ERROR_CORRECTION_EFFICIENCY
                        * entropy(
                            [
                                bit_error_rate / filtered_rate,
                                1 - bit_error_rate / filtered_rate,
                            ],
                            base=2,
                        )
                    )
                    / 3,
                    0.0,
                ]
            )
            print(
                "Result: Depolarizing parameter =",
                depolarizing_parameter,
                ", key rate with conventional PE operator with bit/phase "
                "correlation =",
                conventional_key_rate,
                ", density operator to achieve the conventional rate =",
                conventional_density_operator.value,
                ", key rate with new PEC procedure =",
                new_key_rate,
                ", density operator to achieve the new rate =",
                new_density_operator,
            )
            results.append(
                [
                    depolarizing_parameter,
                    conventional_key_rate,
                    new_key_rate,
                    conventional_problem.value / (np.log(2) * filtered_rate),
                    -new_result.fun,
                ]
            )
        else:
            print(
                "Result: Depolarizing parameter =",
                depolarizing_parameter,
                ", the conventional analysis fails to produce a non-zero "
                "key rate, key rate with new PEC procedure =",
                new_key_rate,
                ", density operator to achieve the new rate =",
                new_density_operator,
            )
            results.append(
                [
                    depolarizing_parameter,
                    0.0,
                    new_key_rate,
                    0.0,
                    -new_result.fun,
                ]
            )

    return results


def save_results(results, output_stem):
    """Save the numerical results as a comma-separated file."""
    csv_path = Path(f"{output_stem}.csv")
    np.savetxt(csv_path, X=np.asarray(results), delimiter=",")


def plot_results(results, output_stem):
    """Plot the calculated key rates and save the figure as SVG."""
    result_array = np.asarray(results)
    x_ticks = np.linspace(0, 0.1, 6)
    y_ticks = np.array([10.0**-exponent for exponent in range(7)])

    plt.rcParams["font.size"] = 14
    figure, axes = plt.subplots()
    axes.set_title("B92 asymptotic key rate")

    axes.set_xlim([0, 0.1])
    axes.set_xticks(x_ticks)
    axes.set_xticklabels([f"{tick}" for tick in x_ticks])
    axes.set_xlabel("Depolarizing parameter", size=14)

    axes.set_ylim([10.0**-6, 1.0])
    axes.set_yticks(y_ticks)
    axes.set_yticklabels([f"{tick:.1e}" for tick in y_ticks])
    axes.set_ylabel("Key rate", size=14)
    axes.set_yscale("log")

    axes.scatter(
        result_array[:, 0], result_array[:, 2], marker=".", s=20, label="New"
    )
    axes.scatter(
        result_array[:, 0],
        result_array[:, 1],
        marker=".",
        s=20,
        label="Conventional",
    )
    axes.legend(loc="lower right")

    svg_path = Path(f"{output_stem}.svg")
    figure.savefig(svg_path, format="svg")
    print("Printed at", svg_path)
    plt.close(figure)


def main():
    """Run the analysis and write its tabular and graphical outputs."""
    results = calculate_key_rates()
    save_results(results, OUTPUT_STEM)
    plot_results(results, OUTPUT_STEM)


if __name__ == "__main__":
    main()
