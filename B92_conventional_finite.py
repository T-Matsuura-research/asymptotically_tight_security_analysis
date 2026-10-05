#!/usr/bin/env python3
"""Compute conventional finite-key rates for the B92 QKD protocol.

The script accepts a depolarizing parameter and an output path stem. It writes
the numerical results to <file_name>.csv and a plot to <file_name>.png.

Runtime dependencies: cvxpy, matplotlib, Mosek, mpmath, numpy, and scipy.
Mosek must be installed and licensed separately.
"""

from __future__ import annotations

import argparse
import logging
from csv import writer
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
from mpmath import superfac
from scipy.optimize import brentq, minimize
from scipy.special import xlogy
from scipy.stats import entropy


LOGGER = logging.getLogger(__name__)

# Protocol and security parameters.
MINUS_LOG_SECURITY = 50 # Correctness/secrecy
MINUS_LOG_PEC_EPSILON = (
    2 * MINUS_LOG_SECURITY + 1
) + np.log2(3)
ERROR_CORRECTION_EFFICIENCY = 1.0
ALPHA = 0.38
BETA = float(np.sqrt(1.0 - ALPHA**2))
MEASUREMENT_GAIN = 1.0

# Each point uses equal probabilities for signal, test, and trash rounds.
SIGNAL_PROBABILITY = 1.0 / 3.0
TEST_PROBABILITY = 1.0 / 3.0
TRASH_PROBABILITY = 1.0 / 3.0

# Alternative sweep retained for reproducing longer runs:
# LOG_TOTAL_ROUNDS = np.linspace(6.5, 11.5, 150)
LOG_TOTAL_ROUNDS = np.linspace(5.0, 10.5, 100)
NUMBER_OF_OUTCOMES = 4
INITIAL_GAP_DEVIATION = 0.25
GAP_DEVIATION_SCALE = 0.45
REFERENCE_TOTAL_ROUNDS = 10.0 ** LOG_TOTAL_ROUNDS[0]
OPTIMIZATION_FAILURE_VALUE = 1.0
NELDER_MEAD_TOLERANCE = 1e-8
MAX_ITERATIONS = 1000


@dataclass(frozen=True)
class FiniteSizeParameters:
    """Round-dependent rates and confidence bounds."""

    total_rounds: float
    minus_log_epsilon: float
    filter_rate: float
    bit_error_rate: float
    bit_error_bound: float
    bounds: Tuple[float, float, float, float, float]


@dataclass(frozen=True)
class RoundResult:
    """One row of finite-key-rate results."""

    log10_total_rounds: float
    key_rate: float
    bit_error_entropy: float
    phase_error_entropy: float
    empirical_probabilities: np.ndarray
    affine_coefficients: np.ndarray

    def as_row(self) -> Sequence[float]:
        return (
            self.log10_total_rounds,
            self.key_rate,
            self.bit_error_entropy,
            self.phase_error_entropy,
            *self.empirical_probabilities,
            *self.affine_coefficients,
        )


def conditional_entropy(probabilities: cp.Expression) -> cp.Expression:
    """Return H(phase | bit) for a five-outcome probability vector."""

    no_bit_total = probabilities[0] + probabilities[1]
    bit_total = probabilities[2] + probabilities[3]
    terms = (
        -cp.rel_entr(probabilities[0], no_bit_total)
        - cp.rel_entr(probabilities[1], no_bit_total)
        - cp.rel_entr(probabilities[2], bit_total)
        - cp.rel_entr(probabilities[3], bit_total)
    )
    return terms / np.log(2.0)


class PhaseErrorModel:
    """CVXPY model used by the scalar phase-error minimization."""

    def __init__(self) -> None:
        self.protocol_parameters = cp.Parameter(3, nonneg=True)
        alpha = self.protocol_parameters[0]
        beta = self.protocol_parameters[1]
        gain = self.protocol_parameters[2]

        filter_matrix = SIGNAL_PROBABILITY * gain * cp.bmat(
            [
                [alpha**2, 0, 0, 0],
                [0, beta**2, 0, 0],
                [0, 0, alpha**2, 0],
                [0, 0, 0, beta**2],
            ]
        )
        bit_error_matrix = SIGNAL_PROBABILITY * gain * cp.bmat(
            [
                [alpha**2 / 2.0, -alpha * beta / 2.0, 0, 0],
                [-alpha * beta / 2.0, beta**2 / 2.0, 0, 0],
                [0, 0, alpha**2 / 2.0, -alpha * beta / 2.0],
                [0, 0, -alpha * beta / 2.0, beta**2 / 2.0],
            ]
        )

        self.state_elements = cp.Variable(6)
        self.state = cp.bmat(
            [
                [
                    self.state_elements[0],
                    self.state_elements[4],
                    0,
                    0,
                ],
                [
                    self.state_elements[4],
                    self.state_elements[1],
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    self.state_elements[2],
                    self.state_elements[5],
                ],
                [
                    0,
                    0,
                    self.state_elements[5],
                    self.state_elements[3],
                ],
            ]
        )

        no_phase_no_bit = cp.trace(
            (filter_matrix[:2, :2] - bit_error_matrix[:2, :2])
            @ self.state[:2, :2]
        )
        phase_no_bit = cp.trace(
            (filter_matrix[2:, 2:] - bit_error_matrix[2:, 2:])
            @ self.state[2:, 2:]
        )
        no_phase_bit = cp.trace(
            bit_error_matrix[:2, :2] @ self.state[:2, :2]
        )
        phase_bit = cp.trace(
            bit_error_matrix[2:, 2:] @ self.state[2:, 2:]
        )
        rejected = 1.0 - cp.trace(filter_matrix @ self.state)
        self.model_probabilities = cp.hstack(
            [
                no_phase_no_bit,
                phase_no_bit,
                no_phase_bit,
                phase_bit,
                rejected,
            ]
        )

        # [lower filter rate, upper minus rate, upper bit-error rate,
        #  rejected-event rate, upper filter rate]
        self.bounds = cp.Parameter(5, nonneg=True)
        physical_constraints = [
            cp.sum(self.state_elements[:4]) == 1.0,
            cp.geo_mean(self.state_elements[:2])
            >= cp.abs(self.state_elements[4]),
            cp.geo_mean(self.state_elements[2:4])
            >= cp.abs(self.state_elements[5]),
            self.model_probabilities[4] <= 1.0 - self.bounds[0],
            TRASH_PROBABILITY * cp.sum(self.state_elements[1:3])
            <= self.bounds[1],
            cp.sum(self.model_probabilities[2:4]) <= self.bounds[2],
            self.model_probabilities[4] >= 1.0 - self.bounds[4],
        ]

        # [P(no phase, no bit), P(phase, no bit), P(no phase, bit),
        #  P(phase, bit), P(rejected)]
        self.empirical_probabilities = cp.Variable(5, nonneg=True)
        self.auxiliary_constraints = [
            self.empirical_probabilities == self.model_probabilities
        ] + physical_constraints
        self.auxiliary_problem = cp.Problem(
            cp.Maximize(
                conditional_entropy(self.empirical_probabilities)
            ),
            self.auxiliary_constraints,
        )

        self.linear_gap = cp.Parameter(nonneg=True)
        self.projection_scale = cp.Parameter(nonneg=True)
        self.linear_coefficients = cp.Parameter(5)
        information_projection = cp.sum(
            cp.rel_entr(
                self.empirical_probabilities,
                self.model_probabilities,
            )
        )
        self.information_projection_problem = cp.Problem(
            cp.Minimize(
                cp.multiply(
                    self.projection_scale,
                    information_projection,
                )
            ),
            physical_constraints
            + [
                self.linear_coefficients @ self.empirical_probabilities
                >= self.linear_gap,
                cp.sum(self.empirical_probabilities) == 1.0,
                self.empirical_probabilities[4] == self.bounds[3],
            ],
        )
        self.phase_entropy_problem = cp.Problem(
            cp.Maximize(
                conditional_entropy(self.empirical_probabilities)
            ),
            [
                self.linear_coefficients @ self.empirical_probabilities
                <= self.linear_gap,
                cp.sum(self.empirical_probabilities) == 1.0,
                self.empirical_probabilities[4] == self.bounds[3],
            ],
        )

        self.protocol_parameters.value = np.array(
            [ALPHA, BETA, MEASUREMENT_GAIN]
        )
        self._solutions: Dict[
            float, Tuple[np.ndarray, np.ndarray]
        ] = {}

    def prepare_round(
        self,
        finite_size: FiniteSizeParameters,
    ) -> None:
        """Set parameters that remain fixed during scalar minimization."""

        self.bounds.value = np.asarray(finite_size.bounds)
        self.projection_scale.value = finite_size.total_rounds / (
            finite_size.minus_log_epsilon * np.log(2.0)
        )
        self._solutions.clear()

    def loss(self, deviation: np.ndarray) -> float:
        """Evaluate the worst-case phase entropy at one gap deviation."""

        deviation_value = float(np.asarray(deviation).reshape(-1)[0])
        self.auxiliary_problem.solve(
            verbose=False,
            solver=cp.MOSEK,
            accept_unknown=True,
        )
        if self.auxiliary_problem.status != cp.OPTIMAL:
            LOGGER.warning(
                "Auxiliary optimization failed with status %s",
                self.auxiliary_problem.status,
            )
            return OPTIMIZATION_FAILURE_VALUE

        self.linear_coefficients.value = (
            self.auxiliary_constraints[0].dual_value
        )
        # Retained diagnostic from the research version:
        # LOGGER.debug("Dual values: %s", self.linear_coefficients.value)
        base_gap = float(
            (self.linear_coefficients @ self.empirical_probabilities).value
        )
        self.linear_gap.value = base_gap * (1.0 + deviation_value)

        self.information_projection_problem.solve(
            verbose=False,
            solver=cp.MOSEK,
            accept_unknown=True,
            warm_start=True,
            # Earlier iterative-solver experiment:
            # eps=1e-7,
            # use_indirect=False,
        )
        projection_status = self.information_projection_problem.status
        projection_value = self.information_projection_problem.value
        if (
            projection_status != cp.OPTIMAL #, cp.OPTIMAL_INACCURATE,}
            or projection_value is None
            or projection_value < 1.0
        ):
            LOGGER.warning(
                "Information projection failed: status=%s, value=%s",
                projection_status,
                projection_value,
            )
            return OPTIMIZATION_FAILURE_VALUE
        #if projection_status == cp.OPTIMAL_INACCURATE:
        #    LOGGER.warning(
        #        "Information projection is inaccurate: %s",
        #        projection_value,
        #    )

        self.phase_entropy_problem.solve(
            verbose=False,
            solver=cp.MOSEK,
            mosek_params={
                "MSK_IPAR_INTPNT_SOLVE_FORM": "MSK_SOLVE_DUAL"
            },
            accept_unknown=True,
        )
        if self.phase_entropy_problem.status != cp.OPTIMAL:
            LOGGER.warning(
                "Phase-entropy optimization failed with status %s",
                self.phase_entropy_problem.status,
            )
            return OPTIMIZATION_FAILURE_VALUE

        empirical = np.asarray(
            self.empirical_probabilities.value
        ).copy()
        coefficients = np.asarray(
            self.linear_coefficients.value
        ).copy()
        self._solutions[deviation_value] = (
            empirical,
            coefficients,
        )
        return float(self.phase_entropy_problem.value)

    def solution_at(
        self,
        deviation: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return the stored solver values nearest an evaluated deviation."""

        if not self._solutions:
            return None
        closest = min(
            self._solutions,
            key=lambda value: abs(value - deviation),
        )
        return self._solutions[closest]


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


def finite_size_parameters(
    dep: float,
    total_rounds: float,
) -> FiniteSizeParameters:
    """Calculate observed rates and finite-size confidence bounds."""

    polynomial_factor = (
        (total_rounds + NUMBER_OF_OUTCOMES - 1) ** (15.0 / 2.0)
        / np.sqrt(
            2.0
            * np.pi
            * (NUMBER_OF_OUTCOMES / np.e**2) ** NUMBER_OF_OUTCOMES
        )
        / float(superfac(NUMBER_OF_OUTCOMES - 1))
    )
    # Five allocated failures: upper-minus, upper-bit, lower-filter,
    # upper-filter, and empirical-phase estimates.
    minus_log_epsilon = (
        MINUS_LOG_PEC_EPSILON
        + np.log2(polynomial_factor)
        + np.log2(5.0)
    )

    base_filter_rate = MEASUREMENT_GAIN * (
        2.0 * ALPHA**2 * BETA**2 * (1.0 - dep) + dep / 2.0
    )
    filter_rate = SIGNAL_PROBABILITY * base_filter_rate
    test_filter_rate = TEST_PROBABILITY * base_filter_rate
    bit_error_rate = (
        TEST_PROBABILITY * MEASUREMENT_GAIN * dep / 4.0
    )
    minus_rate = TRASH_PROBABILITY * ALPHA**2

    pre_sifting_bit_bound = _upper_reverse_kl_probability(
        bit_error_rate / test_filter_rate,
        (MINUS_LOG_SECURITY + 1.0)
        / (total_rounds * (filter_rate + test_filter_rate)),
    ) # MINUS_LOG_SECURITY + 1.0 = -\log(\varepsilon_{cor} / 2)
    bit_error_bound = (
        pre_sifting_bit_bound * (filter_rate + test_filter_rate)
        - bit_error_rate
    ) / filter_rate

    observed_minus_rate = _upper_forward_kl_probability(
        minus_rate,
        MINUS_LOG_PEC_EPSILON / total_rounds,
    )
    upper_minus_rate = _upper_reverse_kl_probability(
        observed_minus_rate,
        minus_log_epsilon / total_rounds,
    )
    upper_bit_error_rate = _upper_reverse_kl_probability(
        bit_error_rate,
        minus_log_epsilon / total_rounds,
    )
    lower_filter_rate = 1.0 - _upper_reverse_kl_probability(
        1.0 - filter_rate,
        minus_log_epsilon / total_rounds,
    )
    upper_filter_rate = _upper_reverse_kl_probability(
        filter_rate,
        minus_log_epsilon / total_rounds,
    )

    return FiniteSizeParameters(
        total_rounds=total_rounds,
        minus_log_epsilon=minus_log_epsilon,
        filter_rate=filter_rate,
        bit_error_rate=bit_error_rate,
        bit_error_bound=bit_error_bound,
        bounds=(
            lower_filter_rate,
            upper_minus_rate,
            upper_bit_error_rate,
            1.0 - filter_rate,
            upper_filter_rate,
        ),
    )


def save_plot(
    results: Sequence[RoundResult],
    output_path: Path,
) -> None:
    """Save a log-scale key-rate plot."""

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    axis.set_title("B92 finite key rate")
    axis.set_xlim(LOG_TOTAL_ROUNDS[0], LOG_TOTAL_ROUNDS[-1])
    axis.set_xlabel(r"$\log_{10}$(number of communication rounds)")
    axis.set_ylim(1e-6, 1.0)
    axis.set_yticks(
        [10.0**-exponent for exponent in range(7)],
        [f"{10.0**-exponent:.1e}" for exponent in range(7)],
    )
    axis.set_ylabel("Key rate")
    axis.set_yscale("log")

    positive_results = [
        result for result in results if result.key_rate > 0.0
    ]
    if positive_results:
        axis.scatter(
            [
                result.log10_total_rounds
                for result in positive_results
            ],
            [result.key_rate for result in positive_results],
            marker=".",
            s=20,
            label="Conventional analysis",
        )
        axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    LOGGER.info("Wrote %s", output_path)


def run_simulation(
    dep: float,
    csv_path: Path,
) -> Sequence[RoundResult]:
    """Run the finite-key sweep and write each completed point to CSV."""

    model = PhaseErrorModel()
    results = []
    header = [
        "log10_total_rounds",
        "key_rate",
        "bit_error_entropy",
        "phase_error_entropy",
        "empirical_no_phase_no_bit",
        "empirical_phase_no_bit",
        "empirical_no_phase_bit",
        "empirical_phase_bit",
        "empirical_rejected",
        "affine_coefficient_0",
        "affine_coefficient_1",
        "affine_coefficient_2",
        "affine_coefficient_3",
        "affine_coefficient_4",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        csv_writer = writer(output_file)
        csv_writer.writerow(header)

        for log_total_rounds in LOG_TOTAL_ROUNDS:
            total_rounds = 10.0**log_total_rounds
            initial_deviation = np.array([min(INITIAL_GAP_DEVIATION, GAP_DEVIATION_SCALE + np.sqrt(REFERENCE_TOTAL_ROUNDS / total_rounds))])
            finite_size = finite_size_parameters(dep, total_rounds)
            LOGGER.info(
                "N=10^%.4f: actual bit-error rate=%.6g, "
                "upper bound=%.6g, absolute upper bound=%.6g",
                log_total_rounds,
                finite_size.bit_error_rate / finite_size.filter_rate,
                finite_size.bit_error_bound,
                finite_size.bit_error_bound * finite_size.filter_rate,
            )
            LOGGER.debug("Finite-size parameters: %s", finite_size)

            model.prepare_round(finite_size)
            optimization = minimize(
                model.loss,
                initial_deviation,
                method="Nelder-Mead",
                options={
                    "maxiter": MAX_ITERATIONS,
                    "xatol": NELDER_MEAD_TOLERANCE,
                    "fatol": NELDER_MEAD_TOLERANCE,
                },
            )
            solution = model.solution_at(float(optimization.x[0]))
            if not optimization.success or solution is None:
                LOGGER.warning(
                    "Optimization failed at log10(N)=%.4f: %s",
                    log_total_rounds,
                    optimization.message,
                )
                continue

            empirical_probabilities, affine_coefficients = solution
            bit_error_entropy = float(
                entropy(
                    [
                        finite_size.bit_error_bound,
                        1.0 - finite_size.bit_error_bound,
                    ],
                    base=2,
                )
            )
            phase_error_entropy = (
                float(optimization.fun) / finite_size.filter_rate
            )
            key_rate = max(
                0.0,
                finite_size.filter_rate
                * (
                    1.0
                    - ERROR_CORRECTION_EFFICIENCY
                    * bit_error_entropy
                )
                - float(optimization.fun)
                - (
                    MINUS_LOG_PEC_EPSILON
                    + MINUS_LOG_SECURITY
                    + 1.0
                )
                / total_rounds,
            ) # # MINUS_LOG_SECURITY + 1.0 = -\log(\varepsilon_{cor} / 2)
            result = RoundResult(
                log10_total_rounds=float(log_total_rounds),
                key_rate=key_rate,
                bit_error_entropy=(
                    ERROR_CORRECTION_EFFICIENCY
                    * bit_error_entropy
                ),
                phase_error_entropy=phase_error_entropy,
                empirical_probabilities=empirical_probabilities,
                affine_coefficients=affine_coefficients,
            )
            csv_writer.writerow(result.as_row())
            output_file.flush()
            results.append(result)
            #initial_deviation = np.asarray(optimization.x).copy()

            LOGGER.info(
                "N=10^%.4f: key rate=%.6g, bit entropy=%.6g, "
                "phase entropy=%.6g, gap deviation=%s",
                log_total_rounds,
                key_rate,
                bit_error_entropy,
                phase_error_entropy,
                optimization.x,
            )
            LOGGER.debug(
                "Worst-case empirical probabilities: %s",
                empirical_probabilities,
            )
            LOGGER.debug(
                "Affine-region coefficients: %s",
                affine_coefficients,
            )

    return results


def parse_arguments(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
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
    """Run the conventional finite-key calculation."""

    if not 0.0 < dep < 1.0:
        raise ValueError("dep must be strictly between 0 and 1.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    output_stem = Path(file_name)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(f"{output_stem}.csv")
    png_path = Path(f"{output_stem}.png")

    try:
        results = run_simulation(dep, csv_path)
    except (cp.error.SolverError, RuntimeError) as error:
        LOGGER.error("%s", error)
        return 1
    if not results:
        LOGGER.error("No optimization point converged; no plot was made.")
        return 1

    save_plot(results, png_path)
    return 0


if __name__ == "__main__":
    arguments = parse_arguments()
    raise SystemExit(main(arguments.dep, arguments.file_name))
