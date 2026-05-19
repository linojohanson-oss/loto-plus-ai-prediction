# backtesting_comparativo.py
# ------------------------------------------------------------
# BACKTESTING COMPARATIVO REAL
#
# Compara:
#   1. Modelo híbrido
#   2. Random baseline
#
# MISMAS condiciones:
# - misma ventana temporal
# - misma cantidad de tickets
# - mismos sorteos
#
# Objetivo:
# detectar si el híbrido supera al azar.
#
# Exporta:
#   backtesting_comparativo_Tradicional.xlsx
#
# Uso:
#   python backtesting_comparativo.py
#
# Opcional:
#   python backtesting_comparativo.py Match
# ------------------------------------------------------------

import sys
import random
import numpy as np
import pandas as pd


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

WINDOW_TRAIN = 500

N_PREDICTIONS = 12

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
# HYBRID MODEL
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
# HYBRID PREDICTION
# =========================
def generate_hybrid_prediction(score):

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


def generate_hybrid_multiple(score):

    out = []

    while len(out) < N_PREDICTIONS:

        pred = generate_hybrid_prediction(score)

        if pred not in out:
            out.append(pred)

    return out


# =========================
# RANDOM PREDICTION
# =========================
def generate_random_prediction():

    nums = random.sample(
        range(1, MAX_NUM + 1),
        6
    )

    return sorted(nums)


def generate_random_multiple():

    out = []

    while len(out) < N_PREDICTIONS:

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
def run_backtest(df):

    rows = []

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

        # =====================
        # HYBRID
        # =====================
        score = compute_scores(train_df)

        hybrid_preds = generate_hybrid_multiple(score)

        best_hybrid = 0

        for p in hybrid_preds:

            hits = count_hits(p, actual)

            if hits > best_hybrid:
                best_hybrid = hits

        # =====================
        # RANDOM
        # =====================
        random_preds = generate_random_multiple()

        best_random = 0

        for p in random_preds:

            hits = count_hits(p, actual)

            if hits > best_random:
                best_random = hits

        rows.append({
            "sorteo": int(test_row["sorteo"]),
            "real": "-".join(
                f"{x:02d}" for x in actual
            ),
            "hybrid_hits": best_hybrid,
            "random_hits": best_random,
            "hybrid_win": int(best_hybrid > best_random),
            "random_win": int(best_random > best_hybrid),
            "tie": int(best_hybrid == best_random),
        })

        print(
            f"Sorteo {int(test_row['sorteo'])} | "
            f"Hybrid={best_hybrid} | "
            f"Random={best_random}"
        )

    return pd.DataFrame(rows)


# =========================
# METRICS
# =========================
def build_metrics(df):

    metrics = []

    hybrid_avg = df["hybrid_hits"].mean()
    random_avg = df["random_hits"].mean()

    metrics.append({
        "metrica": "total_sorteos",
        "valor": len(df)
    })

    metrics.append({
        "metrica": "hybrid_avg_hits",
        "valor": round(hybrid_avg, 4)
    })

    metrics.append({
        "metrica": "random_avg_hits",
        "valor": round(random_avg, 4)
    })

    metrics.append({
        "metrica": "diff_hybrid_minus_random",
        "valor": round(hybrid_avg - random_avg, 4)
    })

    metrics.append({
        "metrica": "hybrid_win_pct",
        "valor": round(
            df["hybrid_win"].mean() * 100,
            2
        )
    })

    metrics.append({
        "metrica": "random_win_pct",
        "valor": round(
            df["random_win"].mean() * 100,
            2
        )
    })

    metrics.append({
        "metrica": "tie_pct",
        "valor": round(
            df["tie"].mean() * 100,
            2
        )
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

    out = f"backtesting_comparativo_{sheet_name}.xlsx"

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

    print("\nOK ✅ COMPARATIVO FINALIZADO")
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
    print("BACKTESTING COMPARATIVO")
    print("==============================")

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