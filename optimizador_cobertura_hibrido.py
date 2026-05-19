# optimizador_cobertura_hibrido.py
# ------------------------------------------------------------
# OPTIMIZADOR HÍBRIDO DE COBERTURA
#
# Usa:
#   - histórico real
#   - frecuencia
#   - recencia
#   - momentum reciente
#
# PERO:
#   el objetivo principal sigue siendo:
#   - cobertura
#   - diversidad
#   - bajo overlap
#
# NO intenta "predecir".
#
# Usa directamente:
#   loto_plus_incremental.xlsx
#
# Exporta:
#   optimizador_hibrido.xlsx
#
# Uso:
#   python optimizador_cobertura_hibrido.py
#
# Opcional:
#   python optimizador_cobertura_hibrido.py 5
# ------------------------------------------------------------

import sys
import random
import itertools
import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
INPUT_XLSX = "loto_plus_incremental.xlsx"

SHEET = "Tradicional"

NUM_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]

MAX_NUM = 45

NUMBERS_PER_TICKET = 6

DEFAULT_TICKETS = 5

N_CANDIDATES = 5000

WINDOW_RECENT = 50

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# =========================
# LOAD DATA
# =========================
def load_draws():

    df = pd.read_excel(
        INPUT_XLSX,
        sheet_name=SHEET
    )

    for c in NUM_COLS:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        ).astype("Int64")

    df = df[
        df[NUM_COLS].notna().sum(axis=1) == 6
    ].copy()

    return df.reset_index(drop=True)


# =========================
# BUILD WEIGHTS
# =========================
def build_weights(df):

    rows = df[NUM_COLS].values.tolist()

    # =====================
    # global frequency
    # =====================
    freq = np.zeros(MAX_NUM + 1)

    for row in rows:

        for n in row:
            freq[int(n)] += 1

    # =====================
    # recent momentum
    # =====================
    recent = rows[-WINDOW_RECENT:]

    momentum = np.zeros(MAX_NUM + 1)

    for i, row in enumerate(recent):

        weight = 1 + (i / len(recent))

        for n in row:
            momentum[int(n)] += weight

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

    momentum_norm = (
        momentum - momentum.min()
    ) / (
        momentum.max() - momentum.min() + 1e-9
    )

    recency_norm = (
        recency - recency.min()
    ) / (
        recency.max() - recency.min() + 1e-9
    )

    # =====================
    # hybrid score
    # =====================
    score = (
        0.45 * freq_norm +
        0.35 * momentum_norm +
        0.20 * recency_norm
    )

    return score


# =========================
# GENERATE TICKET
# =========================
def generate_ticket(weights):

    probs = np.array(weights[1:])

    probs = np.clip(probs, 0, None)

    probs = probs / probs.sum()

    nums = np.random.choice(
        range(1, MAX_NUM + 1),
        size=NUMBERS_PER_TICKET,
        replace=False,
        p=probs
    )

    return sorted(nums.tolist())


# =========================
# HELPERS
# =========================
def ticket_sum(ticket):

    return sum(ticket)


def overlap(a, b):

    return len(
        set(a).intersection(set(b))
    )


def has_long_sequence(ticket):

    seq = 1

    for i in range(1, len(ticket)):

        if ticket[i] == ticket[i - 1] + 1:

            seq += 1

            if seq >= 4:
                return True

        else:
            seq = 1

    return False


def valid_ticket(ticket):

    # secuencias largas
    if has_long_sequence(ticket):
        return False

    # suma razonable
    s = ticket_sum(ticket)

    if s < 70 or s > 210:
        return False

    # pares/impares
    odds = sum(n % 2 for n in ticket)

    if odds in [0, 6]:
        return False

    return True


# =========================
# PORTFOLIO SCORE
# =========================
def portfolio_score(tickets, weights):

    # =====================
    # coverage
    # =====================
    all_nums = set()

    for t in tickets:
        all_nums.update(t)

    coverage = len(all_nums)

    # =====================
    # overlap penalty
    # =====================
    overlap_penalty = 0

    for a, b in itertools.combinations(tickets, 2):

        ov = overlap(a, b)

        overlap_penalty += ov ** 2

    # =====================
    # weighted quality
    # =====================
    quality = 0

    for t in tickets:

        for n in t:
            quality += weights[n]

    # =====================
    # diversity sums
    # =====================
    sums = [ticket_sum(t) for t in tickets]

    diversity = np.std(sums)

    # =====================
    # final score
    # =====================
    score = (
        coverage * 10
        + quality * 25
        + diversity
        - overlap_penalty * 5
    )

    return score


# =========================
# GENERATE PORTFOLIO
# =========================
def generate_portfolio(
    n_tickets,
    weights
):

    tickets = []

    tries = 0

    while len(tickets) < n_tickets:

        tries += 1

        if tries > 100000:
            break

        t = generate_ticket(weights)

        if not valid_ticket(t):
            continue

        if t in tickets:
            continue

        # evitar overlap fuerte
        bad = False

        for existing in tickets:

            if overlap(t, existing) >= 4:
                bad = True
                break

        if bad:
            continue

        tickets.append(t)

    return tickets


# =========================
# OPTIMIZE
# =========================
def optimize(
    n_tickets,
    weights
):

    best_score = -1e9

    best_portfolio = None

    for i in range(N_CANDIDATES):

        portfolio = generate_portfolio(
            n_tickets,
            weights
        )

        if len(portfolio) != n_tickets:
            continue

        score = portfolio_score(
            portfolio,
            weights
        )

        if score > best_score:

            best_score = score

            best_portfolio = portfolio

        if (i + 1) % 500 == 0:

            print(
                f"Iteración {i + 1}/{N_CANDIDATES} | "
                f"Best={best_score:.2f}"
            )

    return best_portfolio, best_score


# =========================
# EXPORT
# =========================
def export_excel(
    portfolio,
    score,
    weights
):

    rows = []

    for i, t in enumerate(portfolio, 1):

        rows.append({
            "ticket": i,
            "numeros": "-".join(
                f"{x:02d}" for x in t
            ),
            "score_ticket": round(
                sum(weights[n] for n in t),
                4
            ),
            "suma": sum(t),
        })

    df = pd.DataFrame(rows)

    # cobertura
    all_nums = set()

    for t in portfolio:
        all_nums.update(t)

    summary = pd.DataFrame([
        {
            "metric": "portfolio_score",
            "value": round(score, 4)
        },
        {
            "metric": "coverage",
            "value": len(all_nums)
        },
    ])

    with pd.ExcelWriter(
        "optimizador_hibrido.xlsx",
        engine="openpyxl"
    ) as wr:

        df.to_excel(
            wr,
            sheet_name="Tickets",
            index=False
        )

        summary.to_excel(
            wr,
            sheet_name="Resumen",
            index=False
        )

    print("\nOK ✅ OPTIMIZADOR HÍBRIDO")
    print("Archivo: optimizador_hibrido.xlsx")


# =========================
# MAIN
# =========================
def main():

    n_tickets = (
        int(sys.argv[1])
        if len(sys.argv) >= 2
        else DEFAULT_TICKETS
    )

    print("\n==============================")
    print("OPTIMIZADOR HÍBRIDO")
    print("==============================")

    print(f"Tickets: {n_tickets}")

    df = load_draws()

    print(f"Sorteos cargados: {len(df)}")

    weights = build_weights(df)

    portfolio, score = optimize(
        n_tickets,
        weights
    )

    print("\nMEJOR PORTAFOLIO:\n")

    for i, t in enumerate(portfolio, 1):

        print(
            f"{i}. "
            + " ".join(f"{x:02d}" for x in t)
        )

    print(f"\nPortfolio score: {score:.2f}")

    export_excel(
        portfolio,
        score,
        weights
    )


if __name__ == "__main__":
    main()