# backtesting_dl_montecarlo.py
# ------------------------------------------------------------
# MONTE CARLO VALIDATION
# DEEP LEARNING vs RANDOM
#
# Objetivo:
# validar si el edge del modelo
# sobrevive múltiples corridas.
#
# Ejecuta:
# - múltiples seeds
# - DL-like vs Random
# - calcula medias y estabilidad
#
# Exporta:
#   montecarlo_dl_vs_random_Tradicional.xlsx
#
# Uso:
#   python backtesting_dl_montecarlo.py
#
# Opcional:
#   python backtesting_dl_montecarlo.py Match
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

N_SIMULATIONS = 25


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
# DL FEATURES
# =========================
def build_probabilities(train_df):

    rows = train_df[NUM_COLS].values.tolist()

    freq = np.zeros(MAX_NUM + 1)

    # recent weighted
    recent_rows = rows[-ROLLING:]

    for i, row in enumerate(recent_rows):

        weight = 1 + (i / len(recent_rows))

        for n in row:
            freq[int(n)] += weight

    # long memory
    for row in rows:

        for n in row:
            freq[int(n)] += 0.15

    # recency
    recency = np.zeros(MAX_NUM + 1)

    for num in range(1, MAX_NUM + 1):

        delay = len(rows)

        for i in range(len(rows) - 1, -1, -1):

            if num in rows[i]:

                delay = len(rows) - i
                break

        recency[num] = delay

    # normalize
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

    score = (
        0.70 * freq_norm +
        0.30 * rec_norm
    )

    return score


# =========================
# GENERATORS
# =========================
def generate_dl(score):

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

        pred = generate_dl(score)

        if pred not in out:
            out.append(pred)

    return out


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
# SINGLE SIMULATION
# =========================
def run_single_simulation(df, seed):

    random.seed(seed)
    np.random.seed(seed)

    dl_hits_all = []
    rnd_hits_all = []

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

        # DL-like
        score = build_probabilities(train_df)

        dl_preds = generate_dl_multiple(score)

        best_dl = 0

        for p in dl_preds:

            hits = count_hits(p, actual)

            if hits > best_dl:
                best_dl = hits

        # Random
        rnd_preds = generate_random_multiple()

        best_rnd = 0

        for p in rnd_preds:

            hits = count_hits(p, actual)

            if hits > best_rnd:
                best_rnd = hits

        dl_hits_all.append(best_dl)
        rnd_hits_all.append(best_rnd)

    return {
        "seed": seed,
        "dl_avg_hits": np.mean(dl_hits_all),
        "random_avg_hits": np.mean(rnd_hits_all),
        "diff": np.mean(dl_hits_all) - np.mean(rnd_hits_all),
        "dl_win_pct": (
            np.mean(
                np.array(dl_hits_all) >
                np.array(rnd_hits_all)
            ) * 100
        ),
    }


# =========================
# MONTE CARLO
# =========================
def run_montecarlo(df):

    rows = []

    for sim in range(N_SIMULATIONS):

        seed = 1000 + sim

        print(
            f"\nSIMULATION {sim + 1}/{N_SIMULATIONS} "
            f"| seed={seed}"
        )

        res = run_single_simulation(df, seed)

        rows.append(res)

        print(
            f"DL={res['dl_avg_hits']:.4f} | "
            f"RND={res['random_avg_hits']:.4f} | "
            f"DIFF={res['diff']:.4f}"
        )

    return pd.DataFrame(rows)


# =========================
# SUMMARY
# =========================
def build_summary(df):

    return pd.DataFrame([
        {
            "metric": "mean_dl_hits",
            "value": round(df["dl_avg_hits"].mean(), 4)
        },
        {
            "metric": "mean_random_hits",
            "value": round(df["random_avg_hits"].mean(), 4)
        },
        {
            "metric": "mean_diff",
            "value": round(df["diff"].mean(), 4)
        },
        {
            "metric": "std_diff",
            "value": round(df["diff"].std(), 4)
        },
        {
            "metric": "positive_diff_pct",
            "value": round(
                (df["diff"] > 0).mean() * 100,
                2
            )
        },
    ])


# =========================
# EXPORT
# =========================
def export_excel(
    sim_df,
    summary_df,
    sheet_name
):

    out = (
        f"montecarlo_dl_vs_random_"
        f"{sheet_name}.xlsx"
    )

    with pd.ExcelWriter(
        out,
        engine="openpyxl"
    ) as wr:

        sim_df.to_excel(
            wr,
            sheet_name="Simulaciones",
            index=False
        )

        summary_df.to_excel(
            wr,
            sheet_name="Resumen",
            index=False
        )

    print("\nOK ✅ MONTE CARLO FINALIZADO")
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
    print("MONTE CARLO DL vs RANDOM")
    print("==============================")

    print(f"Modalidad: {sheet}")

    df = load_draws(sheet)

    print(f"Sorteos: {len(df)}")

    sim_df = run_montecarlo(df)

    summary_df = build_summary(sim_df)

    export_excel(
        sim_df,
        summary_df,
        sheet
    )


if __name__ == "__main__":
    main()