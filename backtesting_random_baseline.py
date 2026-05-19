# backtesting_random_baseline.py
# ------------------------------------------------------------
# BASELINE RANDOM — LOTO PLUS
#
# Objetivo:
# Comparar el modelo híbrido contra azar puro.
#
# Genera:
# - 12 combinaciones RANDOM por sorteo
# - toma la mejor
# - mide aciertos
#
# Exporta:
#   baseline_random_Tradicional.xlsx
#
# Uso:
#   python backtesting_random_baseline.py
#
# Opcional:
#   python backtesting_random_baseline.py Match
# ------------------------------------------------------------

import sys
import random
from typing import List

import pandas as pd
import numpy as np


# =========================
# CONFIG
# =========================
INPUT_XLSX = "loto_plus_incremental.xlsx"

VALID_SHEETS = [
    "Tradicional",
    "Match",
    "Desquite",
    "Sale o Sale",
]

NUM_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]

MAX_NUM = 45

N_RANDOM_PREDICTIONS = 12

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# =========================
# LOAD
# =========================
def load_draws(sheet_name):

    df = pd.read_excel(
        INPUT_XLSX,
        sheet_name=sheet_name
    )

    df["sorteo"] = pd.to_numeric(
        df["sorteo"],
        errors="coerce"
    ).astype(int)

    for c in NUM_COLS:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        ).astype("Int64")

    df = df[
        df[NUM_COLS].notna().sum(axis=1) == 6
    ].copy()

    df = df.sort_values(
        "sorteo"
    ).reset_index(drop=True)

    return df


# =========================
# RANDOM GENERATION
# =========================
def generate_random_prediction():

    nums = random.sample(
        range(1, MAX_NUM + 1),
        6
    )

    return sorted(nums)


def generate_multiple_random():

    out = []

    while len(out) < N_RANDOM_PREDICTIONS:

        pred = generate_random_prediction()

        if pred not in out:
            out.append(pred)

    return out


# =========================
# EVALUATION
# =========================
def count_hits(pred, actual):

    return len(
        set(pred).intersection(set(actual))
    )


# =========================
# BACKTEST
# =========================
def run_random_backtest(df):

    results = []

    for i in range(len(df)):

        row = df.iloc[i]

        actual = [
            int(row[c])
            for c in NUM_COLS
        ]

        preds = generate_multiple_random()

        best_hits = 0
        best_pred = None

        for p in preds:

            hits = count_hits(p, actual)

            if hits > best_hits:
                best_hits = hits
                best_pred = p

        results.append({
            "sorteo": int(row["sorteo"]),
            "real": "-".join(
                f"{x:02d}" for x in actual
            ),
            "mejor_random": "-".join(
                f"{x:02d}" for x in best_pred
            ),
            "aciertos": best_hits,
        })

        print(
            f"Sorteo {int(row['sorteo'])} | "
            f"Random hits={best_hits}"
        )

    return pd.DataFrame(results)


# =========================
# METRICS
# =========================
def build_metrics(results_df):

    total = len(results_df)

    avg_hits = results_df["aciertos"].mean()

    metrics = []

    metrics.append({
        "metrica": "total_sorteos",
        "valor": total
    })

    metrics.append({
        "metrica": "promedio_aciertos",
        "valor": round(avg_hits, 4)
    })

    for k in range(1, 7):

        pct = (
            (results_df["aciertos"] >= k)
            .mean()
        ) * 100

        metrics.append({
            "metrica": f"pct_{k}_o_mas",
            "valor": round(pct, 2)
        })

    return pd.DataFrame(metrics)


# =========================
# EXPORT
# =========================
def export_excel(
    results_df,
    metrics_df,
    sheet_name
):

    out = f"baseline_random_{sheet_name}.xlsx"

    with pd.ExcelWriter(
        out,
        engine="openpyxl"
    ) as wr:

        results_df.to_excel(
            wr,
            sheet_name="Resultados",
            index=False
        )

        metrics_df.to_excel(
            wr,
            sheet_name="Metricas",
            index=False
        )

    print("\nOK ✅ BASELINE RANDOM FINALIZADO")
    print(f"Archivo: {out}")


# =========================
# MAIN
# =========================
def main():

    sheet = (
        sys.argv[1]
        if len(sys.argv) >= 2
        else "Tradicional"
    )

    if sheet not in VALID_SHEETS:
        raise ValueError(
            f"Modalidad inválida: {sheet}"
        )

    print("\n==============================")
    print("BASELINE RANDOM LOTO PLUS")
    print("==============================")

    print(f"Modalidad: {sheet}")

    df = load_draws(sheet)

    print(f"Sorteos cargados: {len(df)}")

    results_df = run_random_backtest(df)

    metrics_df = build_metrics(results_df)

    export_excel(
        results_df,
        metrics_df,
        sheet
    )


if __name__ == "__main__":
    main()