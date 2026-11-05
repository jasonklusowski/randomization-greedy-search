#!/usr/bin/env python3
"""Compute and check the Figure 1 weights from Theorem 3.1."""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import json

K = 10
RANKS = 20
GAMMA_LABELS = ("0.05", "0.15", "0.333")
PRECISION = 80


def closed_form(q: Decimal, j: int, k: int) -> Decimal:
    """Evaluate the finite sum in Theorem 3.1 at the active precision."""
    total = Decimal(0)
    for i in range(1, k + 1):
        product = Decimal(1)
        for ell in range(i, k + 1):
            product *= q ** (j - ell) - q ** j
        total += (-1) ** (k - i) * product
    return total


def recurrence(q: Decimal, k: int, ranks: int) -> list[Decimal]:
    """Evaluate the positive recurrence in Appendix B without cancellation."""
    previous = [Decimal(0)] * (ranks + 1)
    for _ in range(k):
        current = [Decimal(0)]
        for j in range(1, ranks + 1):
            current.append(
                (1 - q) * q ** (j - 1)
                + q ** j * previous[j]
                + (1 - q ** (j - 1)) * previous[j - 1]
            )
        previous = current
    return previous


def compute() -> tuple[dict[tuple[int, str], list[float]], dict]:
    curves, checks = {}, []
    with localcontext() as context:
        context.prec = PRECISION
        gammas = (Decimal("0.05"), Decimal("0.15"), Decimal(1) / 3)
        tolerance = Decimal("1e-65")
        for index, gamma in enumerate(gammas):
            q = 1 - gamma
            independent = recurrence(q, K, RANKS)
            alpha = -q.ln()
            h = ((K * alpha).exp() - 1).ln() / alpha
            exact_curve, approximation_curve = [], []
            maximum_recurrence_error = Decimal(0)
            previous = Decimal(1)
            for j in range(1, RANKS + 1):
                exact = closed_form(q, j, K)
                error = abs(exact - independent[j])
                maximum_recurrence_error = max(maximum_recurrence_error, error)
                if error > tolerance:
                    raise ArithmeticError("Closed form disagrees with the positive recurrence.")
                if not 0 <= exact <= previous:
                    raise ArithmeticError("Weights violate their range or rank monotonicity.")
                previous = exact
                lower = (1 - q ** K) / (1 - q ** K + q ** (K - j))
                upper = (1 - q ** K) / (1 - q ** K + q ** (K - j + 1))
                if exact < lower - tolerance or exact > upper + tolerance:
                    raise ArithmeticError("A weight violates the logistic bounds in Theorem 3.1.")
                approximation = 1 / (1 + (alpha * (Decimal(j) - h - Decimal("0.5"))).exp())
                exact_curve.append(float(exact))
                approximation_curve.append(float(approximation))
            curves[index, "exact"] = exact_curve
            curves[index, "approximation"] = approximation_curve
            checks.append(dict(gamma=GAMMA_LABELS[index],
                               maximum_closed_form_recurrence_difference=str(maximum_recurrence_error)))
    report = dict(status="passed", k=K, ranks=RANKS, curve_coordinates=2 * len(checks) * RANKS,
                  decimal_precision=PRECISION, logistic_midpoint="h(alpha, k) + 1/2",
                  range_monotonicity_and_logistic_bounds="passed", curves=checks)
    return curves, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Print the numerical checks.")
    parser.parse_args()
    _, report = compute()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
