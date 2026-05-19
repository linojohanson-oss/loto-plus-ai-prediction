# backtesting_loto_plus.py
# ------------------------------------------------------------
# BACKTESTING AUTOMÁTICO — LOTO PLUS
#
# V1:
# - Walk-forward temporal
# - Modelo híbrido estadístico
# - Métricas reales
# - Export Excel
#
# Usa:
#   loto_plus_incremental.xlsx
#
# Genera:
#   backtesting_loto_plus.xlsx
#
# Uso:
#   python backtesting_loto_plus.py
#
# Opcional:
#   python backtesting_loto_plus.py Tradicional
# ------------------------------------------------------------

import sys
import random
from typing import List, Dict

import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
INPUT_XLSX = "loto_plus_incremental.xlsx"
OUTPUT_XLSX = "backtesting_loto_plus.xlsx"

VALID_SHEETS = [
    "Tradicional",
    "Match",
    "Desquite",
    "Sale o Sale",
]

NUM_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]

MAX_NUM = 45

WINDOW_TRAIN = 500
N_PREDICTIONS = 12

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# =========================
# LOAD
# =========================
def load_draws(sheet_name):
    df = pd.read_excel(INPUT_XLSX, sheet_name=sheet_name)

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

    df = df.sort_values("sorteo").reset_index(drop=True)

    return df


# =========================
# STATS
# =========================
def compute_scores(train_df):

    all_nums = []

    for _, row in train_df.iterrows():
        for c in NUM_COLS:
            all_nums.append(int(row[c]))

    freq = pd.Series(all_nums).value_counts()

    freq_arr = np.zeros(MAX_NUM + 1)

    for n in range(1, MAX_NUM + 1):
        freq_arr[n] = freq.get(n, 0)

    # recency
    recency = np.zeros(MAX_NUM + 1)

    rows = train_df[NUM_COLS].values.tolist()

    for num in range(1, MAX_NUM + 1):

        delay = len(rows)

        for i in range(len(rows) - 1, -1, -1):

            if num in rows[i]:
                delay = len(rows) - i
                break

        recency[num] = delay

    # normalize
    freq_norm = (
        freq_arr - freq_arr.min()
    ) / (
        freq_arr.max() - freq_arr.min() + 1e-9
    )

    rec_norm = (
        recency - recency.min()
    ) / (
        recency.max() - recency.min() + 1e-9
    )

    score = (
        0.55 * freq_norm +
        0.45 * rec_norm
    )

    return score


# =========================
# PREDICT
# =========================
def generate_prediction(score):

    weights = np.array(score[1:])

    weights = np.clip(weights, 0, None)

    probs = weights / weights.sum()

    nums = np.random.choice(
        range(1, MAX_NUM + 1),
        size=6,
        replace=False,
        p=probs
    )

    return sorted(nums.tolist())


def generate_multiple_predictions(score):

    out = []

    while len(out) < N_PREDICTIONS:

        pred = generate_prediction(score)

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
def run_backtest(df):

    results = []

    start_idx = WINDOW_TRAIN

    for i in range(start_idx, len(df)):

        train_df = df.iloc[
            i - WINDOW_TRAIN:i
        ].copy()

        test_row = df.iloc[i]

        actual = [
            int(test_row[c])
            for c in NUM_COLS
        ]

        score = compute_scores(train_df)

        preds = generate_multiple_predictions(score)

        best_hits = 0
        best_pred = None

        for p in preds:

            hits = count_hits(p, actual)

            if hits > best_hits:
                best_hits = hits
                best_pred = p

        results.append({
            "sorteo": int(test_row["sorteo"]),
            "real": "-".join(
                f"{x:02d}" for x in actual
            ),
            "mejor_prediccion": "-".join(
                f"{x:02d}" for x in best_pred
            ),
            "aciertos": best_hits,
        })

        print(
            f"Sorteo {int(test_row['sorteo'])} | "
            f"Aciertos={best_hits}"
        )

    return pd.DataFrame(results)


# =========================
# METRICS
# =========================
def build_metrics(df_results):

    total = len(df_results)

    avg_hits = df_results["aciertos"].mean()

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
            (df_results["aciertos"] >= k)
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

    out = OUTPUT_XLSX.replace(
        ".xlsx",
        f"_{sheet_name}.xlsx"
    )

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

    print("\nOK ✅ BACKTEST FINALIZADO")
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
            f"Hoja inválida: {sheet}"
        )

    print("\n========================")
    print("BACKTESTING LOTO PLUS")
    print("========================")

    print(f"Modalidad: {sheet}")

    df = load_draws(sheet)

    print(f"Sorteos cargados: {len(df)}")

    results_df = run_backtest(df)

    metrics_df = build_metrics(results_df)

    export_excel(
        results_df,
        metrics_df,
        sheet
    )


if __name__ == "__main__":
    main()