"""
Load inertia results from a precomputed CSV and print LaTeX tables.

Input (required):
  - results/clustering/inertia_values.csv. Format:
      dataset,<model_1>,<model_2>,...
    An empty cell means that run did not complete.

This script:
- Loads the CSV into memory
- Reports how many datasets each model has results for
- Restricts the comparison to the datasets on which MBA, DBA and every
  CADE-BA-hard-dist and soft-DBA-hard-dist run in GAMMAS are all present, so every
  cell of the table has the same denominator
- Computes "Better (%)" for:
    CADE-BA-hard-dist-gamma-{g} vs MBA
    soft-DBA-hard-dist-gamma-{g} vs DBA
- Prints the LaTeX table, stating the number of datasets in the caption.

Notes
-----
- "Better (%)" = percentage of datasets where soft inertia < baseline inertia
  (ties count as not better).
- Percentages cannot be negative with this definition.
"""

from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "results" / "clustering" / "inertia_values.csv"

# Order in table. gamma = 1.0 is excluded: CADE-BA-hard-dist-gamma-1.0 is missing on
# 40 datasets, and including it would shrink the common dataset set from 87 to 69.
GAMMAS = ["0.1", "0.01", "0.001"]
SOFT_BASES = {"CADE-BA-hard-dist": "MBA", "soft-DBA-hard-dist": "DBA"}


def _parse_float(cell: str) -> float | None:
    s = cell.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_inertia_matrix(
    csv_path: Path,
) -> tuple[list[str], list[str], dict[tuple[str, str], float | None]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header or header[0].strip().lower() != "dataset":
            raise ValueError("CSV must have first column header 'dataset'.")

        models = [h.strip() for h in header[1:] if h.strip()]
        datasets: list[str] = []
        inertias: dict[tuple[str, str], float | None] = {}

        for row in reader:
            if not row:
                continue
            ds = row[0].strip()
            if not ds:
                continue

            datasets.append(ds)

            # Ensure row length matches header; missing trailing cells treated as empty
            values = row[1:] + [""] * max(0, len(models) - (len(row) - 1))
            for model, cell in zip(models, values):
                inertias[(model, ds)] = _parse_float(cell)

    return datasets, models, inertias


def report_coverage(
    inertias: dict[tuple[str, str], float | None],
    datasets: list[str],
    models: list[str],
) -> None:
    print(f"Results coverage ({len(datasets)} datasets in CSV):")
    for model in models:
        missing = [ds for ds in datasets if inertias.get((model, ds)) is None]
        print(f"  {model}: {len(datasets) - len(missing)} present, {len(missing)} missing")


def common_datasets(
    inertias: dict[tuple[str, str], float | None],
    datasets: list[str],
    models: list[str],
) -> list[str]:
    """Return the datasets on which every model in ``models`` has a result."""
    return [
        ds for ds in datasets if all(inertias.get((m, ds)) is not None for m in models)
    ]


def compare_soft_vs_base(
    inertias: dict[tuple[str, str], float | None],
    datasets: list[str],
    soft_base: str,
    base_model: str,
    gammas: list[str],
) -> dict[str, float]:
    results: dict[str, float] = {}

    for g in gammas:
        soft_model = f"{soft_base}-gamma-{g}"

        better = 0
        compared = 0

        for ds in datasets:
            soft_val = inertias.get((soft_model, ds))
            base_val = inertias.get((base_model, ds))

            if soft_val is None or base_val is None:
                continue

            compared += 1
            if soft_val < base_val:
                better += 1

        results[g] = float("nan") if compared == 0 else 100.0 * better / compared

        print(
            f"{soft_model} vs {base_model}: "
            f"{better}/{compared} better = {results[g]:.2f}%"
        )

    return results


def latex_percent(x: float) -> str:
    if x != x:  # NaN
        return "--"
    return f"{x:.1f}\\,\\%"


def print_latex_table(
    mba_results: dict[str, float],
    dba_results: dict[str, float],
    gammas: list[str],
    n_datasets: int,
) -> None:
    print("\nLaTeX table:\n")
    print(r"\begin{table}[t]")
    print(
        r"\caption{Percentage of datasets on which the soft variants achieve a lower "
        r"inertia than the standard variants, over the "
        f"{n_datasets}"
        r" datasets for which all runs completed.}"
    )
    print(r"\label{tab:soft_hard_dist_vs_baselines_by_gamma}")
    print(r"\centering")
    print(r"\begin{tabular}{c|c|c}")
    print(r"\toprule")
    print(r"$\gamma$ & \textbf{CADE-BA vs MBA} & \textbf{Soft-DBA vs DBA} \\")
    print(r" & Better (\%) & Better (\%) \\")
    print(r"\midrule")

    for g in gammas:
        print(
            f"{g} & {latex_percent(mba_results[g])} & "
            f"{latex_percent(dba_results[g])} \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    datasets, models, inertias = load_inertia_matrix(INPUT_CSV)

    table_models = [
        model
        for soft_base, base_model in SOFT_BASES.items()
        for model in [base_model] + [f"{soft_base}-gamma-{g}" for g in GAMMAS]
    ]
    for required in table_models:
        if required not in models:
            raise ValueError(
                f"Required model column '{required}' not found in CSV header."
            )

    report_coverage(inertias, datasets, models)
    common = common_datasets(inertias, datasets, table_models)
    print(f"\nDatasets with all table runs present: {len(common)} of {len(datasets)}\n")

    mba_results = compare_soft_vs_base(
        inertias=inertias,
        datasets=common,
        soft_base="CADE-BA-hard-dist",
        base_model="MBA",
        gammas=GAMMAS,
    )

    dba_results = compare_soft_vs_base(
        inertias=inertias,
        datasets=common,
        soft_base="soft-DBA-hard-dist",
        base_model="DBA",
        gammas=GAMMAS,
    )

    print_latex_table(mba_results, dba_results, GAMMAS, len(common))
