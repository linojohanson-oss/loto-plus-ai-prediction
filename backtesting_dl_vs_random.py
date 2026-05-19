# backtesting_dl_vs_random.py
# ------------------------------------------------------------
# BACKTESTING:
#   DEEP LEARNING vs RANDOM
#
# Evalúa:
#   1. Transformer DL
#   2. Random baseline
#
# Bajo mismas condiciones.
#
# VERSIÓN LIGERA:
# - NO reentrena por cada sorteo
# - usa rolling probabilities simples
# - rápida
#
# Exporta:
#   backtesting_dl_vs_random_Tradicional.xlsx
#
# Uso:
#   python backtesting_dl_vs_random.py
#
# Opcional:
#   python backtesting_dl_vs_random.py Match
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

WINDOW = 80

ROLLING = 20

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
# FEATURES
# =========================
def build_probabilities(train_df):

    rows = train_df[NUM_COLS].values.tolist()

    freq = np.zeros(MAX_NUM + 1)

    # =====================
    # rolling recent weight
    # =====================
    recent_rows = rows[-ROLLING:]

    for i, row in enumerate(recent_rows):

        weight = 1 + (i / len(recent_rows))

        for n in row:
            freq[int(n)] += weight

    # =====================
    # long memory
    # =====================
    for row in rows:

        for n in row:
            freq[int(n)] += 0.15

    # =====================
    # recency
    # =====================
    recency = np.zeros(MAX_NUM + 1)

    for num in range(1, MAX_NUM + 1):

        delay = len(rows)

        for i in range(len(rows) - 1, -1, -1):

            if num in rows[i]:
                delay = len(rows) - i
                break

        recency[num] = delay

    # =====================
    # normalize
    # =====================
    freq_norm = (
        freq - freq.min()
    ) / (
        freq.max() - freq.min() + 1e-9
    )

    rec_norm = (
        recency - recency.min()
    ) / (
        recency.max() - recency.min() + 1e-9
    )

    # =====================
    # hybrid DL-like score
    # =====================
    score = (
        0.70 * freq_norm +
        0.30 * rec_norm
    )

    return score


# =========================
# DL-LIKE PREDICTION
# =========================
def generate_dl_prediction(score):

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


def generate_dl_multiple(score):

    out = []

    while len(out) < N_PREDICTIONS:

        pred = generate_dl_prediction(score)

        if pred not in out:
            out.append(pred)

    return out


# =========================
# RANDOM
# =========================
def generate_random():

    nums = random.sample(
        range(1, MAX_NUM + 1),
        6
    )

    return sorted(nums)


def generate_random_multiple():

    out = []

    while len(out) < N_PREDICTIONS:

        pred = generate_random()

        if pred not in out:
            out.append(pred)

    return out


# =========================
# EVAL
# =========================
def count_hits(pred, actual):

    return len(
        set(pred).intersection(set(actual))
    )


# =========================
# BACKTEST
# =========================
def run_backtest(df):

    rows_out = []

    start_idx = WINDOW

    for i in range(start_idx, len(df)):

        train_df = df.iloc[
            i - WINDOW:i
        ].copy()

        test_row = df.iloc[i]

        actual = [
            int(test_row[c])
            for c in NUM_COLS
        ]

        # =====================
        # DL-like
        # =====================
        score = build_probabilities(train_df)

        dl_preds = generate_dl_multiple(score)

        best_dl = 0

        for p in dl_preds:

            hits = count_hits(p, actual)

            if hits > best_dl:
                best_dl = hits

        # =====================
        # RANDOM
        # =====================
        rnd_preds = generate_random_multiple()

        best_rnd = 0

        for p in rnd_preds:

            hits = count_hits(p, actual)

            if hits > best_rnd:
                best_rnd = hits

        rows_out.append({
            "sorteo": int(test_row["sorteo"]),
            "dl_hits": best_dl,
            "random_hits": best_rnd,
            "dl_win": int(best_dl > best_rnd),
            "random_win": int(best_rnd > best_dl),
            "tie": int(best_dl == best_rnd),
        })

        print(
            f"Sorteo {int(test_row['sorteo'])} | "
            f"DL={best_dl} | "
            f"RND={best_rnd}"
        )

    return pd.DataFrame(rows_out)


# =========================
# METRICS
# =========================
def build_metrics(df):

    dl_avg = df["dl_hits"].mean()

    rnd_avg = df["random_hits"].mean()

    metrics = []

    metrics.append({
        "metrica": "total_sorteos",
        "valor": len(df)
    })

    metrics.append({
        "metrica": "dl_avg_hits",
        "valor": round(dl_avg, 4)
    })

    metrics.append({
        "metrica": "random_avg_hits",
        "valor": round(rnd_avg, 4)
    })

    metrics.append({
        "metrica": "diff_dl_minus_random",
        "valor": round(dl_avg - rnd_avg, 4)
    })

    metrics.append({
        "metrica": "dl_win_pct",
        "valor": round(
            df["dl_win"].mean() * 100,
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

    out = (
        f"backtesting_dl_vs_random_"
        f"{sheet_name}.xlsx"
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

    print("\nOK ✅ DL VS RANDOM FINALIZADO")
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
    print("DL VS RANDOM")
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