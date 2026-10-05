#!/usr/bin/env python3
"""Compute finite-size secret-key rates for the B92 QKD protocol.

The calculation uses PICOS and the QICS solver to evaluate the Renyi-entropy
optimization. It writes the numerical results to CSV and creates a PNG plot.
Run this file with --help to see the available configuration options.

Runtime dependencies: cvxopt, matplotlib, mpmath, numpy, picos, qics, and scipy.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from mpmath import superfac
from scipy.optimize import brentq, minimize
from scipy.special import xlogy
from scipy.stats import entropy


LOGGER = logging.getLogger(__name__)

# Fixed protocol and numerical parameters. Only the depolarizing probability
# and output prefix are user inputs.
LOG10_ROUNDS_MIN = 5.0
LOG10_ROUNDS_MAX = 10.5
NUM_POINTS = 100
RECONCILIATION_EFFICIENCY = 1.0
SIGNAL_AMPLITUDE = 0.38
COMPLEMENTARY_AMPLITUDE = float(np.sqrt(1.0 - SIGNAL_AMPLITUDE**2))
GAIN = 1.0
SECURITY_EXPONENT = 50
TEST_PROBABILITY = 1.0 / 3
TRASH_PROBABILITY = 1.0 / 3
EXTRACT_PROBABILITY = 1.0 - TEST_PROBABILITY - TRASH_PROBABILITY
# A phase-error-correction failure probability epsilon implies
# sqrt(2 epsilon)-secrecy.
SAMPLING_SECURITY_EXPONENT = 2 * SECURITY_EXPONENT + 2
SOLVER = "qics"
MAX_ITERATIONS = 1000
NELDER_MEAD_TOLERANCE = 1e-8


@dataclass
class PhaseErrorContext:
    """Fixed inputs and diagnostic state for one scalar optimization."""

    bounds: Tuple[float, float, float, float]
    filtered_rounds: float
    total_rounds: float
    minus_log_epsilon: float
    density_operators: Dict[float, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class RateResult:
    """One row of the generated results table."""

    log10_total_rounds: float
    key_rate: float
    bit_error_entropy: float
    phase_error_entropy: float
    optimized_renyi_parameter: float
    density_operator: Optional[np.ndarray] = field(default=None, repr=False)

    def as_row(self) -> Tuple[float, float, float, float, float]:
        return (
            self.log10_total_rounds,
            self.key_rate,
            self.bit_error_entropy,
            self.phase_error_entropy,
            self.optimized_renyi_parameter,
        )


def _binary_relative_entropy(
    first_probability: float,
    second_probability: float,
) -> float:
    """Return binary relative entropy D(first || second) in bits."""

    first_complement = 1.0 - first_probability
    second_complement = 1.0 - second_probability
    return float(
        (
            xlogy(
                first_probability,
                first_probability / second_probability,
            )
            + xlogy(
                first_complement,
                first_complement / second_complement,
            )
        )
        / np.log(2.0)
    )


def _upper_forward_kl_probability(
    reference_probability: float,
    threshold: float,
) -> float:
    """Solve D(candidate || reference) = threshold above the reference."""

    return float(
        brentq(
            lambda candidate: _binary_relative_entropy(
                candidate,
                reference_probability,
            )
            - threshold,
            reference_probability,
            1.0,
            xtol=1e-14,
            rtol=1e-12,
        )
    )


def _upper_reverse_kl_probability(
    observed_probability: float,
    threshold: float,
) -> float:
    """Solve D(observed || candidate) = threshold above the observation."""

    return float(
        brentq(
            lambda candidate: _binary_relative_entropy(
                observed_probability,
                candidate,
            )
            - threshold,
            observed_probability,
            np.nextafter(1.0, 0.0),
            xtol=1e-14,
            rtol=1e-12,
        )
    )


def _real_scalar(value: complex, description: str) -> float:
    real_value = np.real_if_close(value)
    if np.iscomplexobj(real_value):
        raise RuntimeError(f"{description} has a non-negligible imaginary part")
    return float(real_value)


def _psd_matrix_log(matrix: np.ndarray) -> np.ndarray:
    """Return log(matrix) on its support, with the convention 0 log(0) = 0."""

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = 1e-12 * scale
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError("Expected a positive-semidefinite matrix")

    log_eigenvalues = np.zeros_like(eigenvalues)
    positive = eigenvalues > tolerance
    log_eigenvalues[positive] = np.log(eigenvalues[positive])
    return eigenvectors @ np.diag(log_eigenvalues) @ eigenvectors.conj().T


def phase_error_objective(variable: np.ndarray, context: PhaseErrorContext) -> float:
    """Evaluate the finite-size phase-error objective."""

    try:
        import picos
    except ImportError as error:
        raise RuntimeError(
            "PICOS and its cvxopt dependency are required for optimization"
        ) from error

    renyi_parameter = float(variable[0])
    if not 0.0 < renyi_parameter < 1.0:
        return 10.0

    alpha = SIGNAL_AMPLITUDE
    beta = COMPLEMENTARY_AMPLITUDE

    identity_2 = picos.Constant("identity_2", np.eye(2))
    trash_observable = picos.Constant(
        "trash_observable",
        TRASH_PROBABILITY
        * np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    test_observable = picos.Constant(
        "test_observable",
        TEST_PROBABILITY
        * GAIN
        * np.array(
            [
                [alpha**2 / 2.0, -alpha * beta / 2.0, 0.0, 0.0],
                [-alpha * beta / 2.0, beta**2 / 2.0, 0.0, 0.0],
                [0.0, 0.0, alpha**2 / 2.0, -alpha * beta / 2.0],
                [0.0, 0.0, -alpha * beta / 2.0, beta**2 / 2.0],
            ]
        ),
    )

    density_operator = picos.HermitianVariable("density_operator", 4)
    auxiliary_operator = picos.HermitianVariable("auxiliary_operator", 4)
    z_on_b = picos.Constant(
        "z_on_b", np.array([[0.0, 1.0], [1.0, 0.0]])
    ) @ identity_2
    filter_operator = picos.Constant(
        "filter_operator",
        np.sqrt(GAIN * EXTRACT_PROBABILITY)
        * np.diag([alpha, beta, alpha, beta]),
    )
    filtered_operator = filter_operator * density_operator * filter_operator
    zeros = picos.Constant("zeros", np.zeros((8, 1)))
    classical_quantum_state = picos.block(
        [
            [
                picos.Constant(
                    "plus_projector", [[0.5, 0.0], [0.0, 0.0]]
                )
                @ filtered_operator
                + picos.Constant(
                    "minus_projector", [[0.0, 0.0], [0.0, 0.5]]
                )
                @ (z_on_b * filtered_operator * z_on_b),
                zeros,
            ],
            [zeros.T, 1.0 - picos.trace(filtered_operator)],
        ]
    )
    auxiliary_state = picos.block(
        [
            [identity_2 @ auxiliary_operator, zeros],
            [zeros.T, 1.0 - picos.trace(auxiliary_operator)],
        ]
    )

    problem = picos.Problem()
    renyi_entropy = picos.renyientr(
        classical_quantum_state,
        auxiliary_state,
        1.0 - renyi_parameter,
    )
    problem.set_objective("min", renyi_entropy)
    problem.add_constraint(density_operator >> 0)
    problem.add_constraint(auxiliary_operator >> 0)
    problem.add_constraint(picos.trace(density_operator) == 1.0)
    problem.add_constraint(picos.trace(auxiliary_operator) <= 1.0)
    problem.add_constraint(picos.trace(filtered_operator) <= context.bounds[0])
    problem.add_constraint(
        picos.trace(density_operator * trash_observable) <= context.bounds[1]
    )
    problem.add_constraint(
        picos.trace(density_operator * test_observable) <= context.bounds[2]
    )
    problem.add_constraint(picos.trace(filtered_operator) >= context.bounds[3])

    problem.solve(solver=SOLVER, primals=None, duals=None)
    if problem.status != "optimal":
        LOGGER.warning("Convex optimization status: %s", problem.status)
        return 100.0

    phase_entropy = -float(renyi_entropy) / np.log(2)
    context.density_operators[renyi_parameter] = np.array(density_operator)
    de_finetti_penalty = (
        18.0 * np.log2(context.filtered_rounds + 1)
        + context.minus_log_epsilon / renyi_parameter
    ) / context.total_rounds
    return phase_entropy + de_finetti_penalty


def _entropy_statistics(
    unnormalized_filtered_state: np.ndarray,
    z_on_a: np.ndarray,
) -> Tuple[float, float]:
    averaged_state = (
        unnormalized_filtered_state
        + z_on_a @ unnormalized_filtered_state @ z_on_a
    ) / 2.0
    log_half_state = _psd_matrix_log(unnormalized_filtered_state / 2.0)
    log_averaged_state = _psd_matrix_log(averaged_state)

    first_log_difference = log_half_state - log_averaged_state
    second_log_difference = (
        log_half_state - z_on_a @ log_averaged_state @ z_on_a
    )
    minus_conditional_entropy = _real_scalar(
        (
            np.trace(unnormalized_filtered_state @ first_log_difference)
            + np.trace(unnormalized_filtered_state @ second_log_difference)
        )
        / (2.0 * np.log(2.0)),
        "conditional entropy",
    )

    transformed_state = z_on_a @ unnormalized_filtered_state @ z_on_a
    transformed_log_difference = (
        z_on_a @ log_half_state @ z_on_a - log_averaged_state
    )
    entropy_variance = _real_scalar(
        (
            np.trace(
                unnormalized_filtered_state
                @ first_log_difference
                @ first_log_difference
            )
            + np.trace(
                transformed_state
                @ transformed_log_difference
                @ transformed_log_difference
            )
        )
        / (2.0 * np.log(2.0) ** 2)
        - minus_conditional_entropy**2,
        "entropy variance",
    )
    return minus_conditional_entropy, entropy_variance


def calculate_rate(
    total_rounds: float,
    depolarizing_probability: float,
) -> RateResult:
    """Calculate one B92 key-rate point."""

    alpha = SIGNAL_AMPLITUDE
    beta = COMPLEMENTARY_AMPLITUDE

    filtered_rate = GAIN * (
        2.0
        * alpha**2
        * beta**2
        * (1.0 - depolarizing_probability)
        + depolarizing_probability / 2.0
    )
    bit_error_rate = GAIN * depolarizing_probability / 4.0
    minus_rate = alpha**2

    # Length of sifted keys, successful filtering at test, and number of bit errors at test
    filtered_rounds = total_rounds * EXTRACT_PROBABILITY * filtered_rate
    success_rounds = total_rounds * TEST_PROBABILITY * filtered_rate
    bit_errs = total_rounds * TEST_PROBABILITY * bit_error_rate

    dimension = 4
    prefactor = (
        (total_rounds + dimension - 1.0) ** (15.0 / 2.0)
        / np.sqrt(2.0 * np.pi * (dimension / np.e**2) ** dimension)
        / float(superfac(dimension - 1))
    )
    minus_log_epsilon = (
        SAMPLING_SECURITY_EXPONENT + np.log2(prefactor) + np.log2(5)
    )

    estimated_bit_error = _upper_reverse_kl_probability(
        bit_errs / success_rounds,
        (SECURITY_EXPONENT + 1) / (filtered_rounds + success_rounds),
    ) * (1 + success_rounds / filtered_rounds) - bit_errs / filtered_rounds

    p_minus = TRASH_PROBABILITY * minus_rate
    observed_minus = _upper_forward_kl_probability(
        p_minus,
        SAMPLING_SECURITY_EXPONENT / total_rounds,
    )
    upper_minus = _upper_reverse_kl_probability(
        observed_minus,
        minus_log_epsilon / total_rounds,
    )
    upper_bit = _upper_reverse_kl_probability(
        TEST_PROBABILITY * bit_error_rate,
        minus_log_epsilon / total_rounds,
    )
    lower_filtered = 1.0 - _upper_reverse_kl_probability(
        1.0 - EXTRACT_PROBABILITY * filtered_rate,
        minus_log_epsilon / total_rounds,
    )
    upper_filtered = _upper_reverse_kl_probability(
        EXTRACT_PROBABILITY * filtered_rate,
        minus_log_epsilon / total_rounds,
    )

    expected_state = (1.0 - depolarizing_probability) * np.array(
        [
            [beta**2, alpha * beta, 0.0, 0.0],
            [alpha * beta, alpha**2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ) + depolarizing_probability * np.diag(
        [beta**2, alpha**2, alpha**2, beta**2]
    ) / 2.0
    filter_operator = np.sqrt(GAIN) * np.diag(
        [alpha, beta, alpha, beta]
    )
    unnormalized_filtered_state = (
        EXTRACT_PROBABILITY
        * filter_operator
        @ expected_state
        @ filter_operator
    )
    z_on_a = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    minus_conditional_entropy, entropy_variance = _entropy_statistics(
        unnormalized_filtered_state,
        z_on_a,
    )
    if entropy_variance <= 0.0:
        raise RuntimeError(
            f"Entropy variance must be positive; received {entropy_variance}"
        )

    initial_renyi_parameter = min(
        np.sqrt(
            2.0 * np.log2(np.e) * minus_log_epsilon / (total_rounds * entropy_variance)
        ),
        1.0 - 1e-3,
    )
    LOGGER.info(
        "N=10^%.4f: initial Renyi parameter=%.6g, "
        "conditional entropy=%.6g, entropy variance=%.6g",
        np.log10(total_rounds),
        initial_renyi_parameter,
        -minus_conditional_entropy,
        entropy_variance,
    )

    context = PhaseErrorContext(
        bounds=(upper_filtered, upper_minus, upper_bit, lower_filtered),
        filtered_rounds=filtered_rounds,
        total_rounds=total_rounds,
        minus_log_epsilon=minus_log_epsilon,
    )
    optimization = minimize(
        phase_error_objective,
        x0=np.array([initial_renyi_parameter]),
        args=(context,),
        method="Nelder-Mead",
        options={
            "maxiter": MAX_ITERATIONS,
            "xatol": NELDER_MEAD_TOLERANCE,
            "fatol": NELDER_MEAD_TOLERANCE,
        },
    )

    bit_entropy = float(
        entropy([estimated_bit_error, 1.0 - estimated_bit_error], base=2)
    )
    if not optimization.success or not context.density_operators:
        LOGGER.warning(
            "Scalar optimization failed for N=10^%.4f: %s",
            np.log10(total_rounds),
            optimization.message,
        )
        return RateResult(
            log10_total_rounds=float(np.log10(total_rounds)),
            key_rate=0.0,
            bit_error_entropy=bit_entropy,
            phase_error_entropy=0.0,
            optimized_renyi_parameter=float("nan"),
        )

    optimized_parameter = float(optimization.x[0])
    closest_parameter = min(
        context.density_operators,
        key=lambda value: abs(value - optimized_parameter),
    )
    optimized_density_operator = context.density_operators[closest_parameter]
    phase_entropy = float(optimization.fun) / (
        EXTRACT_PROBABILITY * filtered_rate
    )
    key_rate = max(
        EXTRACT_PROBABILITY
        * filtered_rate
        * (1.0 - RECONCILIATION_EFFICIENCY * bit_entropy)
        - float(optimization.fun) - (SECURITY_EXPONENT + 1) / (EXTRACT_PROBABILITY * filtered_rate * total_rounds),
        0.0,
    )
    LOGGER.info(
        "N=10^%.4f: key rate=%.6g, bit entropy=%.6g, "
        "phase entropy=%.6g, optimized Renyi parameter=%.6g",
        np.log10(total_rounds),
        key_rate,
        bit_entropy,
        phase_entropy,
        optimized_parameter,
    )
    LOGGER.debug("Optimized density operator:\n%s", optimized_density_operator)
    return RateResult(
        log10_total_rounds=float(np.log10(total_rounds)),
        key_rate=key_rate,
        bit_error_entropy=bit_entropy,
        phase_error_entropy=phase_entropy,
        optimized_renyi_parameter=optimized_parameter,
        density_operator=optimized_density_operator,
    )


def run_simulation(
    depolarizing_probability: float,
) -> Sequence[RateResult]:
    """Run the fixed sweep for one depolarizing probability."""

    if not 0.0 <= depolarizing_probability <= 1.0:
        raise ValueError("depolarizing_probability must be in [0, 1]")
    log10_round_counts = np.linspace(
        LOG10_ROUNDS_MIN,
        LOG10_ROUNDS_MAX,
        NUM_POINTS,
    )
    return [
        calculate_rate(10.0**log_count, depolarizing_probability)
        for log_count in log10_round_counts
    ]


def save_results(
    results: Sequence[RateResult],
    output_prefix: Path,
) -> Tuple[Path, Path]:
    """Write numerical results and a key-rate plot."""

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(f"{output_prefix}.csv")
    png_path = Path(f"{output_prefix}.png")
    table = np.asarray([result.as_row() for result in results], dtype=float)
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header=(
            "log10_total_rounds,key_rate,bit_error_entropy,"
            "phase_error_entropy,optimized_renyi_parameter"
        ),
        comments="",
    )

    figure, axis = plt.subplots()
    axis.set_title("B92 key rate with finite-size analysis")
    axis.set_xlabel(r"$\log_{10}$(number of communication rounds)")
    axis.set_ylabel("Key rate")
    axis.set_yscale("log")
    axis.set_ylim(1e-6, 1.0)
    axis.set_yticks([10.0**-power for power in range(7)])

    positive_results = [result for result in results if result.key_rate > 0.0]
    if positive_results:
        axis.scatter(
            [result.log10_total_rounds for result in positive_results],
            [result.key_rate for result in positive_results],
            marker=".",
            s=20,
            label="Finite-size analysis",
        )
        axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(png_path, dpi=150)
    plt.close(figure)

    LOGGER.info("Wrote %s", csv_path)
    LOGGER.info("Wrote %s", png_path)
    return csv_path, png_path


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the script's two public input parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dep",
        type=float,
        help="Depolarizing parameter (strictly between 0 and 1).",
    )
    parser.add_argument(
        "file_name",
        help="Output path stem; '.csv' and '.png' are appended.",
    )
    return parser.parse_args(argv)


def main(dep: float, file_name: str) -> int:
    """Run the finite-key calculation for one depolarizing parameter."""
    if not 0.0 < dep < 1.0:
        raise ValueError("dep must be strictly between 0 and 1.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        results = run_simulation(dep)
    except RuntimeError as error:
        LOGGER.error("%s", error)
        return 1
    save_results(results, Path(file_name))
    return 0


if __name__ == "__main__":
    arguments = parse_arguments()
    raise SystemExit(main(arguments.dep, arguments.file_name))
