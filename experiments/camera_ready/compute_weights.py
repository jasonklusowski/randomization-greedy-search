#!/usr/bin/env python3
"""Compute Figure 1 from Theorem 3.1 and check the published curve coordinates.

The default command writes new theory CSV and validation JSON files.  --check
performs the same calculations without writing files.  The original curve CSV,
its provenance, and the manuscript figures are never changed.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ORIGINAL = RESULTS / "figure1_original_curves.csv"
K = 10
RANKS = 20
PRECISION = 80
PDF_TOLERANCE = 5e-9


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


def compute() -> tuple[list[dict], dict]:
    rows, checks = [], []
    with ORIGINAL.open(newline="") as handle:
        original = list(csv.DictReader(handle))
    if len(original) != 120:
        raise ValueError("Expected all 120 original curve coordinates.")
    observed = {(int(row["gamma_index"]), row["variant"],
                 int(row["point_index"])): row for row in original}
    expected = {(index, variant, j) for index in range(3)
                for variant in ("exact", "approximation")
                for j in range(1, RANKS + 1)}
    if set(observed) != expected:
        raise ValueError("The original curve coordinates are incomplete or duplicated.")

    with localcontext() as context:
        context.prec = PRECISION
        gammas = (Decimal("0.05"), Decimal("0.15"), Decimal(1) / 3)
        for index, gamma in enumerate(gammas):
            q = 1 - gamma
            independent = recurrence(q, K, RANKS)
            alpha = -q.ln()
            h = ((K * alpha).exp() - 1).ln() / alpha
            maximum_recurrence_error = Decimal(0)
            errors = {variant: 0.0 for variant in ("exact", "approximation")}
            previous = Decimal(1)
            for j in range(1, RANKS + 1):
                exact = closed_form(q, j, K)
                error = abs(exact - independent[j])
                maximum_recurrence_error = max(maximum_recurrence_error, error)
                if error > Decimal("1e-65"):
                    raise ArithmeticError("Closed form disagrees with the positive recurrence.")
                if not 0 <= exact <= previous:
                    raise ArithmeticError("Weights violate their range or rank monotonicity.")
                previous = exact
                lower = (1 - q ** K) / (1 - q ** K + q ** (K - j))
                upper = (1 - q ** K) / (1 - q ** K + q ** (K - j + 1))
                if exact < lower - Decimal("1e-65") or exact > upper + Decimal("1e-65"):
                    raise ArithmeticError("A weight violates the logistic bounds in Theorem 3.1.")
                approximation = 1 / (1 + (alpha * (Decimal(j) - h - Decimal("0.5"))).exp())
                for variant, weight in (("exact", exact), ("approximation", approximation)):
                    source = observed[index, variant, j]
                    if abs(float(source["j"]) - j) > 3e-8:
                        raise ValueError("An original curve coordinate has an unexpected rank.")
                    difference = abs(float(weight) - float(source["weight"]))
                    errors[variant] = max(errors[variant], difference)
                    if difference > PDF_TOLERANCE:
                        raise ArithmeticError("Theoretical curves disagree with the original PDF coordinates.")
                    rows.append(dict(gamma_index=index, gamma=str(gamma),
                                     display_gamma=("0.05", "0.15", "0.333")[index],
                                     variant=variant, point_index=j, j=j, weight=str(weight)))
            checks.append(dict(gamma_index=index, gamma=str(gamma),
                               maximum_closed_form_recurrence_difference=str(maximum_recurrence_error),
                               maximum_original_weight_difference=errors,
                               range_monotonicity_and_logistic_bounds="passed"))
    report = dict(status="passed", k=K, ranks=RANKS, curve_coordinates=120,
                  decimal_precision=PRECISION, pdf_weight_tolerance=PDF_TOLERANCE,
                  original_csv="experiments/camera_ready/results/figure1_original_curves.csv",
                  original_csv_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
                  formula="Theorem 3.1 finite sum, with q = 1 - gamma",
                  independent_check="Positive recurrence in Appendix B",
                  logistic_midpoint="h(alpha, k) + 1/2",
                  note="Small differences from the original coordinates reflect PDF serialization rounding.",
                  curves=checks)
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate without writing files.")
    parser.add_argument("--csv", type=Path, default=RESULTS / "figure1_theory_curves.csv")
    parser.add_argument("--report", type=Path, default=RESULTS / "figure1_theory_validation.json")
    args = parser.parse_args()
    rows, report = compute()
    if not args.check:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report["theory_csv_sha256"] = hashlib.sha256(args.csv.read_bytes()).hexdigest()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
